r"""Neural network modules"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # if gpu is to be used


def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    """Creates a multi-layer perceptron (MLP)."""
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class HeteroVisionConv(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads=4):
        super(HeteroVisionConv, self).__init__()
        self.conv = nn.ModuleDict({'gt': dglnn.GATConv((in_feats['gt'], in_feats['agent']),
                                                       int(out_feats / num_heads), num_heads,
                                                       residual=True, allow_zero_in_degree=True, activation=nn.ReLU()),
                                   'uav': dglnn.GATConv((in_feats['uav'], in_feats['agent']),
                                                        int(out_feats / num_heads), num_heads,
                                                        residual=True, allow_zero_in_degree=True, activation=nn.ReLU())
                                   })
        self.f_aggr = nn.Linear(2 * out_feats, out_feats)

    def forward(self, g, x):
        h_gts = self.conv['gt'](g['gt'], (x['gt'], x['agent'])).view(g['gt'].num_nodes('agent'), -1)
        # print(f"h_gts.size() = {h_gts.size()}")
        h_uavs = self.conv['uav'](g['uav'], (x['uav'], x['agent'])).view(g['uav'].num_nodes('agent'), -1)
        # print(f"h_uavs.size() = {h_uavs.size()}")
        return torch.relu(self.f_aggr(torch.cat((h_gts, h_uavs), 1)))


class ContinuousMessagePassing(nn.Module):
    def __init__(self, in_feats, out_feats, hidden_size=128, msg_size=64, aggr_type='mean'):
        super(ContinuousMessagePassing, self).__init__()
        self.f_enc = mlp([in_feats, hidden_size, msg_size], output_activation=nn.ReLU)
        self.f_udt = nn.GRUCell(in_feats + msg_size, out_feats)
        self.aggr_type = aggr_type

    def msg_func(self, edges):
        soft_msg = self.f_enc(edges.src['x'])
        return {'m': soft_msg}

    def aggr_func(self, nodes):
        if self.aggr_type == 'mean':
            return {'y': nodes.mailbox['m'].mean(1)}
        elif self.aggr_type == 'max':
            return {'y': nodes.mailbox['m'].max(1)}
        else:
            raise ValueError('Invalid aggregation type for `ContinuousMessagePassing`.')

    def forward(self, g, x, z):
        with g.local_scope():
            g.ndata['x'] = x
            g.update_all(self.msg_func, self.aggr_func)
            h_out = self.f_udt(torch.cat((g.ndata['x'], g.ndata['y']), 1), z)
            return h_out, h_out


class DiscreteMessagePassing(nn.Module):
    def __init__(self, in_feats, out_feats, hidden_size=128, msg_size=64, tau=0.1):
        super(DiscreteMessagePassing, self).__init__()
        self.f_enc = mlp([in_feats, hidden_size, msg_size])
        self.f_dec = mlp([msg_size, hidden_size], output_activation=nn.ReLU)
        self.f_udt = nn.GRUCell(in_feats + hidden_size, out_feats)  # Here, size of GRU hidden states = `out_feats`.
        self.tau = tau  # Non-negative scalar temperature of Gumbel-Softmax

    def msg_func(self, edges):
        logits = self.f_enc(edges.src['x'])
        if self.training:
            return {'m': F.gumbel_softmax(logits, tau=self.tau, hard=False)}
        else:
            d = torch.distributions.OneHotCategorical(logits=logits)
            return {'m': d.sample()}
            # return {'m': F.gumbel_softmax(logits, tau=self.tau, hard=True)}

    def aggr_func(self, nodes):
        return {'y': nodes.mailbox['m'].max(1)[0]}

    def forward(self, g, x, z):
        with g.local_scope():
            g.ndata['x'] = x
            g.update_all(self.msg_func, self.aggr_func)
            # print(f"g.ndata['x'].size() = {g.ndata['x'].size()}, g.ndata['y'].size() = {g.ndata['y'].size()}")
            h_out = self.f_udt(torch.cat((g.ndata['x'], self.f_dec(g.ndata['y'])), 1), z)
            return h_out, h_out


class GraphVisCommNet(nn.Module):
    def __init__(self, in_feats, out_feats, hidden_size={'vis': 64, 'comm': 128}, num_heads=4,
                 is_het_vis=False, is_c_comm=False, num_comm_steps=2):
        super(GraphVisCommNet, self).__init__()
        self.hidden_size = hidden_size  # Size of intermediate hidden features
        self.hidden_size['rnn'] = self.hidden_size['comm']
        self.num_heads = num_heads  # Number of heads when graph attention networks are employed
        self._is_het_vis = is_het_vis  # If hetero vision is applied
        self._is_c_comm = is_c_comm  # If continuous communication among UAV-BSs is applied
        self.num_comm_steps = num_comm_steps  # Number of communication steps among agents

        # Define graph vision layers according to whether other UAVs are included.
        if self._is_het_vis:
            self.tag = 'het_gvis'
            self.vis1 = HeteroVisionConv(in_feats, hidden_size['vis'], num_heads)
            self.vis2 = HeteroVisionConv({'gt': in_feats['gt'], 'uav': in_feats['uav'],
                                          'agent': hidden_size['vis'] + in_feats['agent']},
                                         hidden_size['comm'], num_heads)
        else:
            self.tag = 'gvis'
            self.vis1 = dglnn.GATConv((in_feats['gt'], in_feats['agent']), int(hidden_size['vis'] / num_heads),
                                      num_heads, residual=True, allow_zero_in_degree=True, activation=nn.ReLU())
            self.vis2 = dglnn.GATConv((in_feats['gt'], hidden_size['vis'] + in_feats['agent']),
                                      int(hidden_size['comm'] / num_heads),
                                      num_heads, residual=True, allow_zero_in_degree=True, activation=nn.ReLU())

        # Define communication layer and specify if continuous/discrete communication is used.
        if self._is_c_comm:
            self.tag = self.tag + '_c_comm'
            self.comm = ContinuousMessagePassing(hidden_size['comm'], hidden_size['comm'],
                                                 hidden_size=128, msg_size=64, aggr_type='mean')
        else:
            self.tag = self.tag + '_d_comm'
            self.comm = DiscreteMessagePassing(hidden_size['comm'], hidden_size['comm'],
                                               hidden_size=128, msg_size=64, tau=0.1)
        self.f_out = nn.Linear(hidden_size['comm'], out_feats)

    def forward(self, g, z):
        g = g.to(device)
        h_in = g.ndata['feat']
        if self._is_het_vis:
            g_vis = {'gt': g['gt', 'found-by', 'agent'], 'uav': g['uav', 'close-to', 'agent']}
            h_vis1 = self.vis1(g_vis, h_in)
            h_vis2 = self.vis2(g_vis, {'gt': h_in['gt'], 'uav': h_in['uav'],
                                       'agent': torch.cat((h_in['agent'], h_vis1), 1)})
        else:
            g_vis = g['gt', 'found-by', 'agent']
            h_vis1 = self.vis1(g_vis, (h_in['gt'], h_in['agent'])).view(g.num_nodes('agent'), -1)
            h_vis2 = self.vis2(g_vis, (h_in['gt'], torch.cat((h_vis1, h_in['agent']), 1))).view(g.num_nodes('agent'), -1)

        g_comm = g['agent', 'talks-to', 'agent']
        h = h_vis2
        for l in range(self.num_comm_steps):
            h, z = self.comm(g_comm, h, z)
            h = h.view(g.num_nodes('agent'), self.hidden_size['comm'])
        h_out = self.f_out(h)
        return h_out, z