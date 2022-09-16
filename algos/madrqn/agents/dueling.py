import torch.nn as nn


class DuelingLayer(nn.Module):
    """Dueling architecture"""

    def __init__(self, in_feats, n_actions):
        super(DuelingLayer, self).__init__()

        self.adv_head = nn.Linear(in_feats, n_actions)
        self.v_head = nn.Linear(in_feats, 1)

    def forward(self, x):
        vals = self.v_head(x)
        advs = self.adv_head(x)
        return vals + (advs - advs.mean(-1, keepdims=True))
