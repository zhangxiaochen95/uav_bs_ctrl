import torch as th
import torch.nn as nn
from algos.madrqn.agents.dueling import DuelingLayer


class RnnAgent(nn.Module):
    """Recurrent policy for independent agents"""

    def __init__(self, obs_shape, n_actions, args):
        super(RnnAgent, self).__init__()

        self._n_layers = args.n_layers  # Number of fully-connected layers
        self._hidden_size = args.hidden_size  # Dimension of hidden features

        # Create fully-connected layers as observation encoder.
        layers = [nn.Linear(obs_shape, self._hidden_size), nn.ReLU()]
        for _ in range(self._n_layers - 1):
            layers += [nn.Linear(self._hidden_size, self._hidden_size), nn.ReLU()]
        self.enc = nn.Sequential(*layers)

        self.rnn = nn.GRUCell(self._hidden_size, self._hidden_size)  # Recurrent layer
        if args.dueling:
            self.f_out = DuelingLayer(self._hidden_size, n_actions)
        else:
            self.f_out = nn.Linear(self._hidden_size, n_actions)

    def init_hidden(self):
        """Initializes hidden state for recurrent layers."""
        return th.zeros(1, self._hidden_size)

    def forward(self, obs, h):
        x = self.enc(obs)
        h = self.rnn(x, h)
        x = self.f_out(h)
        return x, h
