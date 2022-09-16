from envs.common import *


class Map(object):
    """Base class of all maps"""

    def __init__(self, range_pos=500, episode_limit=20, dt=10, n_ubs=1, n_gts=1, r_cov=100., n_rbs=1,
                 r_sns=np.inf, r_comm=np.inf, vels=10, n_dirs=4, rew_scale=1.):

        self.range_pos = range_pos  # Range of positions (m)
        self.episode_limit = episode_limit  # Maximum number of timesteps in an episode
        self.dt = dt  # Time length of each step (sec)

        self.n_ubs = n_ubs  # Number of UBSs
        self.n_gts = n_gts  # Number of GTs
        self.r_cov = r_cov  # Range of wireless coverage (m)
        self.n_rbs = n_rbs  # Number of resource blocks

        self.r_sns = r_sns  # Sensing range of nearby GTs (m)
        self.r_comm = r_comm  # Range of multi-agent communication (m)

        self.vels = vels  # Velocities of UBSs (m/sec)
        self.n_dirs = n_dirs  # Number of discrete flying directions

        self.reward_scale_rate = rew_scale  # Scaling factor of rewards

    def get_params(self):
        """Extracts all parameters."""
        return self.__dict__

    def set_positions(self):
        # Uniformly sample positions of UBSs and GTs.
        pos_ubs = select_from_cube(self.n_ubs, 0, self.range_pos, 2)
        pos_gts = select_from_cube(self.n_gts, 0, self.range_pos, 2)
        return dict(ubs=pos_ubs, gt=pos_gts)


class Debug(Map):
    """Deterministic setup for debug"""

    def __init__(self, range_pos=1000, episode_limit=10, dt=10, n_ubs=3, n_gts=4, r_cov=100., n_rbs=1,
                 r_sns=300., r_comm=np.inf, vels=10., n_dirs=4, rew_scale=1.):

        super(Debug, self).__init__(range_pos, episode_limit, dt, n_ubs, n_gts, r_cov, n_rbs,
                                    r_sns, r_comm, vels, n_dirs, rew_scale)

    def set_positions(self):
        pos_ubs = 100 * np.array([[3, 3], [8, 2], [8, 9]], dtype=np.float32)
        pos_gts = 100 * np.array([[3, 4], [4, 2], [3, 1], [6, 9]], dtype=np.float32)
        return dict(ubs=pos_ubs, gt=pos_gts)

# ==================================================================================================================== #
# Setup of experiment 2


class HotSpot(Map):
    """Random hotspot composed of a few GTs"""

    def __init__(self, range_pos=2000, episode_limit=40, dt=20, n_ubs=4, n_gts=4, r_cov=100., n_rbs=1,
                 r_sns=200., r_comm=np.inf, vels=[5, 10], n_dirs=4, rew_scale=10.):
        super(HotSpot, self).__init__(range_pos, episode_limit, dt, n_ubs, n_gts, r_cov, n_rbs,
                                      r_sns, r_comm, vels, n_dirs, rew_scale)

    def set_positions(self):
        min_dist = 200.  # Minimum length between GTs (m)
        pos_ubs = min_dist * select_from_cube(self.n_ubs, 0, self.range_pos // min_dist, 2)  # Positions of UBSs

        range_spot = 1  # Range of hotspot (*min_dist)
        while np.square(range_spot) < self.n_gts: range_spot += 1  # Ensure sufficient area to hold GTs
        pos_spot = min_dist * range_spot * select_from_cube(1, 0, self.range_pos // min_dist // range_spot, 2)
        pos_gts = pos_spot + min_dist * select_from_cube(self.n_gts, 0, range_spot, 2)

        pos_gts = np.clip(pos_gts, 0, self.range_pos)  # Clip positions.
        np.random.shuffle(pos_gts)  # Shuffle order of GTs.
        return dict(ubs=pos_ubs, gt=pos_gts)

# ==================================================================================================================== #
# Setup of experiment 3


# Easy mode
class DenseHotSpot(Map):
    """
    Random hotspot composed of a few groups of GTs
    The growing number of GTs makes it harder to process observations.
    """

    def __init__(self, range_pos=6000, episode_limit=50, dt=40, n_ubs=4, n_grps=10, gts_per_grp=5,
                 r_cov=100., n_rbs=5, r_sns=400., r_comm=np.inf, vels=[5, 10], n_dirs=4, rew_scale=10):

        n_gts = n_grps * gts_per_grp
        super(DenseHotSpot, self).__init__(range_pos, episode_limit, dt, n_ubs, n_gts, r_cov, n_rbs,
                                           r_sns, r_comm, vels, n_dirs, rew_scale)
        self.n_grps = n_grps  # Number of GT groups
        self.gts_per_grp = gts_per_grp  # Population of each group

    def set_positions(self):
        min_dist = 200.  # Minimum length between GT groups (m)
        pos_ubs = min_dist * select_from_cube(self.n_ubs, 0, self.range_pos // min_dist, 2)  # Positions of UBSs

        range_spot = 1  # Range of hotspot (*min_dist)
        while np.square(range_spot) < self.n_grps: range_spot += 1  # Ensure sufficient area to hold GTs
        pos_spot = min_dist * range_spot * select_from_cube(1, 0, self.range_pos // min_dist // range_spot, 2)
        pos_grps = pos_spot + min_dist * select_from_cube(self.n_grps, 0, range_spot, 2)  # Positions of groups
        pos_gts = np.empty((self.n_gts, 2), dtype=np.float32)
        # Position of each GT is randomly drawn from the group center.
        for grp_idx in range(self.n_grps):
            gt_idxes = slice(grp_idx * self.gts_per_grp, (grp_idx + 1) * self.gts_per_grp)
            pos_gts[gt_idxes] = pos_grps[grp_idx] + self.r_cov * (np.random.rand(self.gts_per_grp, 2) - 0.5)

        pos_gts = np.clip(pos_gts, 0, self.range_pos)  # Clip positions.
        np.random.shuffle(pos_gts)  # Shuffle order of GTs.
        return dict(ubs=pos_ubs, gt=pos_gts)


# Hard Mode
class DenseHotSpotV2(Map):
    """Compared with easy mode, finer control is requested."""

    def __init__(self, range_pos=6000., episode_limit=100, dt=10, n_ubs=4, n_gts=100,
                 r_cov=100., n_rbs=10, r_sns=400, r_comm=np.inf, vels=[5., 10.], n_dirs=4, rew_scale=10):
        super(DenseHotSpotV2, self).__init__(range_pos, episode_limit, dt, n_ubs, n_gts, r_cov, n_rbs,
                                             r_sns, r_comm, vels, n_dirs, rew_scale)

    def set_positions(self):
        pos_ubs = 100 * select_from_cube(self.n_ubs, 0, self.range_pos // 100, 2)
        radius_spot = 400  # Half the length of user hotspot (m)
        pos_spot = radius_spot * select_from_cube(1, 1, self.range_pos // radius_spot, 2)
        pos_gts = pos_spot + radius_spot * 2 * (np.random.rand(self.n_gts, 2) - 0.5)
        pos_gts = np.clip(pos_gts, 0, self.range_pos)  # Clip positions.
        np.random.shuffle(pos_gts)  # Shuffle order of GTs.
        return dict(ubs=pos_ubs, gt=pos_gts)

# ==================================================================================================================== #


# Registry of maps
MAPS = {
    'test': Map(),
    'debug': Debug(),

    # Experiment 2
    'inf': HotSpot(),
    'r400': HotSpot(r_comm=400.),
    'r800': HotSpot(r_comm=800.),

    # Experiment 3
    '4ubs': DenseHotSpot(n_ubs=4),
    '6ubs': DenseHotSpot(n_ubs=6),
    '8ubs': DenseHotSpot(n_ubs=8),
}
