r"""Initialization functions of UAV-BSs and GTs"""
import numpy as np
from random import sample, shuffle
import itertools
import math


def uni_select(max_val, n_select, n_dims=2):
    """Randomly selects `n_els` unique elements from a $(0, max_val)^{n_dim}$ box."""
    opts = list(itertools.product(*list(itertools.repeat(np.arange(1, max_val), n_dims))))
    # opts = list(itertools.product(*list(itertools.repeat(np.arange(0, max_val+1), n_dims))))
    return np.array(sample(opts, n_select))


def setup_uavs(n_grids, len_grid, n_uavs):
    # Random distribution of UAV-BSs.
    assert n_uavs <= 4
    avail_spots = [[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]]
    spots = sample(avail_spots, n_uavs)
    p_uavs = n_grids * len_grid * np.array(spots, dtype=np.float64)
    return p_uavs


def setup_gts(n_grids, len_grid, std_dev):
    len_area = n_grids * len_grid  # Length of the area
    n_clts, size_clts = 10, 1

    n_gts = n_clts * size_clts
    spots_gts = len_grid * uni_select(n_grids, n_clts)
    sizes_clts = n_clts * [size_clts]

    # Shuffle the order of GTs in clusters.
    x = list(range(n_gts))
    pos_gts = np.zeros((n_gts, 2), dtype=np.float64)
    for c in range(n_clts):
        samples = sample(x, sizes_clts[c])
        x = list(set(x) - set(samples))
        pos_gts[samples] = spots_gts[c] + std_dev * len_grid * np.random.randn(sizes_clts[c], 2)
    # print(f"pos_gts = {pos_gts}")
    pos_gts = np.clip(pos_gts, 0, len_area)
    return n_gts, pos_gts
