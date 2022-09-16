import torch as th
import torch.nn as nn
import torch.nn.functional as F

import dgl.nn.pytorch as dglnn
import dgl.function as fn
from dgl.nn.functional import edge_softmax

from algos.madrqn.agents.dueling import DuelingLayer


class GnnAgent(nn.Module):
    """Recurrent agents leveraging graph neural networks"""

    def __init__(self, obs_shape, n_actions, args):
        super(GnnAgent, self).__init__()

        self._hidden_size = args.hidden_size  # Dimension of hidden features
        self._comm_protocol = args.c  # Communication protocol among agents

        # Create observation encoder.
        if isinstance(obs_shape, int):
            self.enc = DenseObservationEncoder(obs_shape, args)
        elif isinstance(obs_shape, dict):
            self.enc = GraphObservationEncoder(obs_shape, args)

        # Create module for multi-agent communication (MAC).
        if self._comm_protocol is None:
            self.rnn = nn.GRUCell(self._hidden_size, self._hidden_size)  # Independent agents
        elif self._comm_protocol == 'base':
            self.f_comm = BaseComm(args)  # Basic Communication
        elif self._comm_protocol == 'disc':
            self.f_comm = DiscreteComm(args)  # Discrete communication
        elif self._comm_protocol == 'commnet':
            self.f_comm = CommNet(args)  # CommNet
        elif self._comm_protocol == 'tarmac':
            self.f_comm = TarMAC(args)  # TarMAC
        elif self._comm_protocol == 'econv':
            self.f_comm = EdgeConv(args)
        else:
            raise KeyError("Unsupported communication scheme.")

        if args.dueling:
            self.f_out = DuelingLayer(self._hidden_size, n_actions)
        else:
            self.f_out = nn.Linear(self._hidden_size, n_actions)

    def init_hidden(self):
        return th.zeros(1, self._hidden_size)

    def forward(self, g, h):
        # Proceed local observations.
        x = self.enc(g, g.ndata['feat']).view(g.num_nodes('agent'), -1)
        # Call message-passing among agents over communication graph.
        h = self.f_comm(g['talk'], x, h) if self._comm_protocol is not None else self.rnn(x, h)
        return self.f_out(h), h


# ==================================================================================================================== #
# Observation Encoders

class DenseObservationEncoder(nn.Module):
    """MLP observation encoder"""

    def __init__(self, obs_shape, args):
        super(DenseObservationEncoder, self).__init__()

        self._n_layers = args.n_layers  # Number of fc layers
        self._hidden_size = args.hidden_size

        layers = [nn.Linear(obs_shape, self._hidden_size), nn.ReLU()]
        for l in range(self._n_layers - 1):
            layers += [nn.Linear(self._hidden_size, self._hidden_size), nn.ReLU()]
        self.enc = nn.Sequential(*layers)

    def forward(self, g, x):
        return self.enc(x['agent'])


class GraphObservationEncoder(nn.Module):
    """Observation encoder adopting heterogeneous graph convolution"""

    def __init__(self, obs_shape, args):
        super(GraphObservationEncoder, self).__init__()

        n_heads = args.n_heads  # Number of attention heads in GAT
        out_feats = args.hidden_size  # Overall dimension of outputs
        assert out_feats % n_heads == 0, "out_feats cannot be divided by n_heads in GraphObservationLayer."
        feats_per_head = int(out_feats / n_heads)  # Output dimension of each head

        # Define graph convolution layers (GAT) handling each relation.
        self.f_conv = nn.ModuleDict({
            'seen': dglnn.GATv2Conv((obs_shape['gt'], obs_shape['agent']), feats_per_head, n_heads,
                                    residual=True, allow_zero_in_degree=True, activation=nn.ReLU()),
            'near': dglnn.GATv2Conv((obs_shape['ubs'], obs_shape['agent']), feats_per_head, n_heads,
                                    residual=True, allow_zero_in_degree=True, activation=nn.ReLU())
        })

        self.f_aggr = nn.Sequential(nn.Linear(len(self.f_conv) * out_feats, out_feats), nn.ReLU())  # Aggregator

    def forward(self, g, x):
        # Get features for each relation.
        x_gt = self.f_conv['seen'](g['seen'], (x['gt'], x['agent'])).view(g.num_nodes('agent'), -1)
        x_ubs = self.f_conv['near'](g['near'], (x['ubs'], x['agent'])).view(g.num_nodes('agent'), -1)
        # Combine outputs across relations to get final results.
        x_out = th.cat((x_gt, x_ubs), 1)
        return self.f_aggr(x_out)

# ==================================================================================================================== #
# Communication Protocols


class BaseComm(nn.Module):
    """Base protocol for multi-agent communication"""

    def __init__(self, args):
        super(BaseComm, self).__init__()

        self._hidden_size = args.hidden_size  # Size of hidden states
        self._msg_size = args.msg_size  # Size of messages

        self.f_msg = nn.Linear(self._hidden_size + self._hidden_size, self._msg_size)  # Message function
        self.f_udt = nn.GRUCell(self._hidden_size + self._msg_size, self._hidden_size)  # RNN unit

    def msg_func(self, edges):
        """Encodes messages from local inputs and detached hidden states."""
        msg = self.f_msg(th.cat((edges.src['x'], edges.src['h']), 1))
        return dict(m=msg)

    def aggr_func(self, nodes):
        """Aggregates incoming messages by averaging."""
        aggr_msg = nodes.mailbox['m'].mean(1)
        return dict(c=aggr_msg)

    def forward(self, g, x, h):
        with g.local_scope():
            g.ndata['x'], g.ndata['h'] = x, h.detach()  # Get inputs and the latest hidden states.

            if g.number_of_edges() == 0:
                # When no edge is created, paddle zeros.
                g.dstdata['c'] = th.zeros(x.shape[0], self._hidden_size)
            else:
                # Otherwise, call message passing between nodes.
                g.update_all(self.msg_func, self.aggr_func)

            # Update the hidden states using inputs, aggregated messages and hidden states.
            h = self.f_udt(th.cat((g.ndata['x'], g.ndata['c']), 1), h)
            return h


class DiscreteComm(nn.Module):
    """Discrete Communication"""

    def __init__(self, args):
        super(DiscreteComm, self).__init__()

        self._hidden_size = args.hidden_size  # Size of hidden states
        self._msg_size = args.msg_size  # Size of messages
        # Note: In discrete communication, we use 2 digits to denote 1 bit as either (0, 1) or (1, 0).
        # Therefore, outputs from message encoder take twice the number of digits of continuous counterparts.

        self.f_enc = nn.Linear(self._hidden_size + self._hidden_size, 2 * self._msg_size)  # Message function
        self.f_dec = nn.Linear(2 * self._msg_size, 2 * self._msg_size)  # Decoder of aggregated messages
        self.f_udt = nn.GRUCell(self._hidden_size + 2 * self._msg_size, self._hidden_size)  # RNN unit

    def msg_func(self, edges):
        """Encodes discrete messages from local inputs and detached hidden states."""
        # Get logits from message function.
        logits = self.f_enc(th.cat((edges.src['x'], edges.src['h']), 1))
        # When discrete communication is required,
        # we use Gumbel-Softmax function to sample binary messages while keeping gradients for backpropagation.
        disc_msg = F.gumbel_softmax(logits.view(-1, self._msg_size, 2), tau=0.5, hard=True)
        return dict(m=disc_msg.flatten(1))

    def aggr_func(self, nodes):
        """Aggregates incoming discrete messages by element-wise OR operation."""
        aggr_msg = nodes.mailbox['m'].max(1)[0]
        return dict(c=aggr_msg)

    def forward(self, g, x, h):
        with g.local_scope():
            g.ndata['x'], g.ndata['h'] = x, h.detach()  # Get inputs and the latest hidden states.

            if g.number_of_edges() == 0:
                # When no edge is created, paddle zeros.
                g.dstdata['c'] = th.zeros(x.shape[0], 2 * self._msg_size)
            else:
                # Otherwise, call message passing between nodes.
                g.update_all(self.msg_func, self.aggr_func)

            # Update the hidden states using inputs, aggregated messages and hidden states.
            h = self.f_udt(th.cat((g.ndata['x'], self.f_dec(g.ndata['c'])), 1), h)
            return h


class CommNet(nn.Module):
    """
    Implementation of CommNet from paper "Learning Multiagent Communication with Backpropagation".
    Actually inherited from IC3Net code
    """

    def __init__(self, args):
        super(CommNet, self).__init__()
        self._hidden_size = args.hidden_size  # Size of hidden states
        self._n_rounds = args.n_rounds  # Number of communication rounds

        self.c_mod = nn.Linear(self._hidden_size, self._hidden_size)  # Module to process aggregated messages
        self.f_mod = nn.GRUCell(self._hidden_size, self._hidden_size)  # Module to update hidden states

    def msg_func(self, edges):
        """Message from each agent is its state."""
        return {'m': edges.src['h']}  # Same as dgl.function.copy_u('h', 'm')

    def aggr_func(self, nodes):
        """Each agent averages all incoming messages from neighbours."""
        return {'c': nodes.mailbox['m'].mean(1)}

    def forward(self, g, x, h):
        with g.local_scope():
            g.ndata['x'] = x
            for l in range(self._n_rounds):
                g.ndata['h'] = h.detach()
                if g.number_of_edges() == 0:
                    g.dstdata['c'] = th.zeros(x.shape[0], self._hidden_size)
                else:
                    g.update_all(self.msg_func, self.aggr_func)
                c = self.c_mod(g.ndata['c'])  # Process aggregated messages.
                h = self.f_mod(g.ndata['x'] + c, h)  # Update hidden states using skip connection.
            return h


class TarMAC(nn.Module):
    """TarMAC: Targeted Multi-Agent Communication"""

    def __init__(self, args):
        super(TarMAC, self).__init__()

        self._hidden_size = args.hidden_size  # Size of hidden states
        self._msg_size = args.msg_size  # Size of messages
        self._key_size = args.key_size  # Size of signatures and queries
        self._n_rounds = args.n_rounds  # Number of multi-round communication

        self.f_val = nn.Linear(2 * self._hidden_size, self._msg_size)  # Value function (producing messages)
        self.f_sign = nn.Linear(2 * self._hidden_size, self._key_size)  # Signature function (predicting keys at Tx)
        self.f_que = nn.Linear(2 * self._hidden_size, self._key_size)  # Query function (predicting keys at Rx)
        self.f_udt = nn.GRUCell(self._hidden_size + self._msg_size, self._hidden_size)  # RNN update function

    def forward(self, g, x, h):
        with g.local_scope():
            g.ndata['x'] = x
            for l in range(self._n_rounds):
                g.ndata['h'] = h  # Get the latest hidden states.
                # Build inputs to modules for communication.
                inputs = th.cat((g.srcdata['x'], g.srcdata['h'].detach()), 1)
                # Get signatures, values at source nodes.
                g.srcdata.update(dict(v=self.f_val(inputs), s=self.f_sign(inputs)))
                # Predict queries at destination nodes.
                g.dstdata.update(dict(q=self.f_que(inputs)))

                # Get attention scores on each edge.
                g.apply_edges(fn.u_dot_v('s', 'q', 'e'))  # Dot-product of signature and query
                e = g.edata.pop('e') / self._key_size  # Divide by key-size
                g.edata['a'] = edge_softmax(g, e)  # Normalize attention score by softmax

                # Get aggregated message at destination nodes.
                g.update_all(fn.u_mul_e('v', 'a', 'm'), fn.sum('m', 'c'))  # m = a * v
                c = g.dstdata['c']  # Aggregated messages

                # Update the hidden states using inputs, aggregated messages and hidden states.
                h = self.f_udt(th.cat((x, c), 1), h)
            return h


class EdgeConv(nn.Module):
    def __init__(self, args):
        super(EdgeConv, self).__init__()
        self._hidden_size = args.hidden_size
        self._msg_size = args.msg_size
        self._n_rounds = args.n_rounds

        self.f_msg = nn.Linear(4 * self._hidden_size, self._msg_size)
        self.f_udt = nn.GRUCell(self._hidden_size + self._msg_size, self._hidden_size)

    def msg_func(self, edges):
        msg = self.f_msg(th.cat([edges.src['x'], edges.src['h'], edges.dst['x'], edges.dst['h']], 1))
        return {'m': msg}

    def aggr_func(self, nodes):
        return {'c': nodes.mailbox['m'].mean(1)}

    def forward(self, g, x, h):
        g.ndata['x'] = x
        for l in range(self._n_rounds):
            g.ndata['h'] = h.detach()
            if g.number_of_edges() == 0:
                g.dstdata['c'] = th.zeros(x.shape[0], self._hidden_size)
            else:
                g.update_all(self.msg_func, self.aggr_func)
            h = self.f_udt(th.cat([g.ndata['x'], g.ndata['c']], 1), h)
        return h