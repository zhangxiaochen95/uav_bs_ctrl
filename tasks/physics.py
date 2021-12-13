r"""Components of environment physics"""
import numpy as np


class AirToGroundChannel(object):
    """Air-to-ground (ATG) channel model"""
    def __init__(self, fc=2.4e9, scene='urban'):
        # Determine the scene-specific parameters.
        chan_params = {'suburban': (4.88, 0.43, 0.1, 21), 'urban': (9.61, 0.16, 1, 20),
                       'dense-urban': (12.08, 0.11, 1.6, 23), 'high-rise-urban': (27.23, 0.08, 2.3, 34)}
        self.a, self.b = chan_params[scene][0], chan_params[scene][1]  # Constants for computing p_los
        self.eta_los, self.eta_nlos = chan_params[scene][2], chan_params[scene][3]  # Path loss exponents (LoS/NLoS)
        self.fc = fc  # Carrier frequency

    def compute_chan_gain(self, dist, h):
        # Estimate probability of LoS link emergence.
        p_los = 1 / (1 + self.a * np.exp(-self.b * (np.arctan(h / (dist + 1e-5)) - self.a)))
        # Get direct link distance
        d = np.sqrt(np.square(dist) + np.square(h))
        # Compute free space path loss (FSPL).
        fspl = (4 * np.pi * self.fc * d / (3e8)) ** 2
        # Path loss is the weighted average of LoS and NLoS cases.
        pl = p_los * fspl * 10 ** (self.eta_los / 20) + (1 - p_los) * fspl * 10 ** (self.eta_nlos / 20)  # Path loss
        return 1 / pl


def dbm2w(x):
    """Covert variables' unit from dBm to Watt."""
    return 1e-3 * np.power(10, x / 10)


def jain_fairness_index(x):
    """Computes the Jain's fairness index of entries in given ndarray."""
    if x.size > 0:
        x = np.clip(x, 1e-6, np.inf)
        return np.square(x.sum()) / (x.size * np.square(x).sum())
    else:
        return 1