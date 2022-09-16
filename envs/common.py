import os
import os.path as osp

from random import sample
from itertools import repeat, product
import numpy as np
import pandas as pd


# ==================================================================================================================== #
# Math

def select_from_cube(n_els, min_val, max_val, n_dims,):
    """Selects non-repetitive elements from a cube."""
    legal_points = list(product(*list(repeat(np.arange(min_val, max_val), n_dims))))
    return np.array(sample(legal_points, n_els))


def compute_jain_fairness_index(x):
    """Computes the Jain's fairness index of entries in given ndarray."""
    if x.size > 0:
        x = np.clip(x, 1e-6, np.inf)
        return np.square(x.sum()) / (x.size * np.square(x).sum())
    else:
        return 1

# ==================================================================================================================== #
# Channel models


class AirToGroundChannel(object):
    """Air-to-ground (ATG) channel model"""

    chan_params = {
        'suburban': (4.88, 0.43, 0.1, 21),
        'urban': (9.61, 0.16, 1, 20),
        'dense-urban': (12.08, 0.11, 1.6, 23),
        'high-rise-urban': (27.23, 0.08, 2.3, 34)
    }

    def __init__(self, scene, fc):
        # Determine the scene-specific parameters.
        params = self.chan_params[scene]
        self.a, self.b = params[0], params[1]  # Constants for computing p_los
        self.eta_los, self.eta_nlos = params[2], params[3]  # Path loss exponents (LoS/NLoS)

        self.fc = fc  # Central carrier frequency (Hz)

    def estimate_chan_gain(self, d_level, h_ubs):
        """Estimates the channel gain from horizontal distance."""
        # Estimate probability of LoS link emergence.
        p_los = 1 / (1 + self.a * np.exp(-self.b * (np.arctan(h_ubs / (d_level + 1e-5)) - self.a)))
        # Get direct link distance.
        d = np.sqrt(np.square(d_level) + np.square(h_ubs))
        # Compute free space path loss (FSPL).
        fspl = (4 * np.pi * self.fc * d / 3e8) ** 2
        # Path loss is the weighted average of LoS and NLoS cases.
        pl = p_los * fspl * 10 ** (self.eta_los / 20) + (1 - p_los) * fspl * 10 ** (self.eta_nlos / 20)
        return 1 / pl


# ==================================================================================================================== #
# Functions to plot simple objects


def plot_line(a, b):
    """Plots a line from a to b."""
    assert a.size == b.size
    assert a.ndim == 1
    return [np.linspace(a[d], b[d], 50) for d in range(a.size)]


def plot_circ(x_o, y_o, r):
    """Plots a circle centered at given origin (x_0, y_o) with radius r."""
    t = np.linspace(0, 2 * np.pi, 100)
    x_data, y_data = r * np.cos(t), r * np.sin(t)
    return x_o + x_data, y_o + y_data


def write_to_disk(save_dir, path_ubs, pos_gts, **kwargs):

    # Get length of episode, number of UBSs/GTs.
    ep_len = path_ubs.shape[0]
    n_ubs = path_ubs.shape[1] if (path_ubs.ndim == 3) else 1
    n_gts = pos_gts.shape[0]

    # Write trajectories of UBSs to disk.
    cols = pd.MultiIndex.from_tuples(product([f'UBS-{i}' for i in range(n_ubs)], ['position'], ['x', 'y']))
    # print(f"cols = {cols}")
    # print(f"cols.shape = {cols.shape}, path_ubs.shape = {path_ubs.shape}")
    path_ubs = pd.DataFrame(path_ubs.reshape(ep_len, -1), columns=cols)
    path_ubs.to_csv(osp.join(save_dir, 'path_ubs.csv'))

    # Write positions of GTs to disk.
    pos_gts = pd.DataFrame(pos_gts, columns=['x', 'y'], index=[f'GT-{m}' for m in range(n_gts)])
    pos_gts.to_csv(osp.join(save_dir, 'pos_gts.csv'))

    # Write other provided named series to disk.
    others = pd.DataFrame(kwargs)
    others.to_csv(osp.join(save_dir, 'others.csv'))


if __name__ == '__main__':
    path = '/Users/zhangxiaochen/PycharmProjects/ubs-cov-gmarl-replica/envs/mubs_cov/recorder.py'
    print(osp.split(path))