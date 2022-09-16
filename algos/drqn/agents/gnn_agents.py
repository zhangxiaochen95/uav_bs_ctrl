import torch as th
import torch.nn as nn
import torch.nn.functional as F

import dgl.nn.pytorch as dglnn
import dgl.function as fn


class GnnAgent(nn.Module):
    def __init__(self, obs_shape, n_actions, args):
        super(GnnAgent, self).__init__()
        self._hidden_size = args.hidden_size  # Hidden size

        self._n_heads = args.n_heads
        feats_per_head = self._hidden_size // self._n_heads

        self.enc = dglnn.GATv2Conv((obs_shape['gt'], obs_shape['agent']), feats_per_head, self._n_heads,
                                   residual=True, allow_zero_in_degree=True, activation=nn.ReLU())

        self.rnn = nn.GRUCell(self._hidden_size, self._hidden_size)
        self.f_out = nn.Linear(self._hidden_size, n_actions)

    def init_hidden(self):
        return th.zeros(1, self._hidden_size)

    def forward(self, g, h):
        x = self.enc(g, (g.nodes['gt'].data['feat'], g.nodes['agent'].data['feat'])).flatten(start_dim=1)
        h = self.rnn(x, h)
        x = self.f_out(h)
        return x, h
