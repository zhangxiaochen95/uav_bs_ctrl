import numpy as np

from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
from gym.spaces.dict import Dict

from envs.common import *
from envs.subs_cov.recorder import Recorder


class SingleUbsCoverageEnv:

    unit = 100  # Length of each unit (m)
    h_ubs = 100  # Elevation of UBSs (m)
    p_tx = 1e-3 * np.power(10, 10 / 10)  # Transmit power (Watt)
    n0 = 1e-3 * np.power(10, -170 / 10)  # PSD of noise (Watt/Hz)
    bw = 180e3  # Bandwidth of channels (Hz)
    fc = 2.4e9  # Central carrier frequency (Hz)
    dt = 10  # Length of timesteps (sec)
    scene = 'urban'  # Scene of channel model

    def __init__(self, range_pos=1000, episode_limit=200, n_grps=2, gts_per_grp=1, r_cov=100., n_rbs=10,
                 vels=10, n_dirs=4, record=True):

        self.range_pos = range_pos
        self.episode_limit = episode_limit

        self.n_grps = n_grps
        self.gts_per_grp = gts_per_grp
        self.n_gts = n_grps * gts_per_grp  # Number of GTs

        self.r_cov = r_cov  # Range of wireless coverage (m)
        self.n_rbs = n_rbs  # Number of sub-channels

        self.chan = AirToGroundChannel(self.scene, self.fc)  # Channel model
        g_max = self.chan.estimate_chan_gain(0, self.h_ubs)  # Maximum channel gain
        snr_max = self.p_tx * g_max / (self.n0 * self.bw)  # SNR of optimal link
        self.max_rate = self.bw * np.log2(1 + snr_max) * 1e-6  # Maximum link rate (Mbps)

        # Variables
        self.t = None  # Timer
        self.pos_ubs = np.empty(2, dtype=np.float32)  # Positions of UBS
        self.pos_gts = np.empty((self.n_gts, 2), dtype=np.float32)  # Positions of GTs
        self.d_u2g = np.empty(self.n_gts, dtype=np.float32)  # Distance between UBSs and GTs
        self.prior_gts = np.empty(self.n_gts, dtype=int)  # Priority of GTs for receiving service
        self.sched = np.empty(self.n_gts, dtype=bool)  # Scheduling relations between UBSs and GTs

        self.rate_per_gt = np.empty(self.n_gts, dtype=np.float32)
        self.aver_rate_per_gt = np.empty(self.n_gts, dtype=np.float32)
        self.fair_idx = None
        self.total_throughput = None
        self.global_util = None  # Trade-off between fairness and throughput
        self.avg_global_util = None

        # Obtain possible actions.
        move_amounts = self.dt * np.array(vels).reshape(-1, 1)  # Possible amounts of movement in each timestep
        ang = 2 * np.pi * np.arange(n_dirs) / n_dirs  # Possible flying angles
        move_dirs = np.stack([np.cos(ang), np.sin(ang)]).T
        self.avail_moves = np.concatenate((np.zeros((1, 2)), np.kron(move_amounts, move_dirs)))  # Available moves
        self.n_actions = self.avail_moves.shape[0]  # Number of actions

        self.observation_space = Dict(spaces={
            'agent': Box(-np.inf, np.inf, shape=np.array([self.obs_own_feats_size])),
            'gt': Box(-np.inf, np.inf, shape=np.array(self.obs_gt_feats_size))
        })
        self.action_space = Discrete(self.n_actions)  # Action space of each agent
        self.reward_scale_rate = self.n_grps

        self.ep_ret = 0  # Episode return
        # Tool for recording the episode.
        self.recorder = None
        if record:
            self.recorder = Recorder(self)

    def reset(self):
        self.t = 0
        self.ep_ret = 0
        self.avg_global_util = 0
        self.aver_rate_per_gt = np.zeros(self.n_gts, dtype=np.float32)
        self.total_throughput = 0

        self._set_position()

        self.prior_gts = np.random.permutation(self.n_gts)
        self._transmit_data()  # UBSs provide wireless service for GTs.

        if self.recorder is not None:
            self.recorder.reload()

        return self.get_obs()

    def _set_position(self):
        # self.pos_ubs = select_from_cube(1, 0, self.range_pos, 2).squeeze()
        # self.pos_gts = select_from_cube(self.n_gts, 0, self.range_pos, 2)
        self.pos_ubs = np.array([self.range_pos / 2, self.range_pos / 2], dtype=np.float32)

        # Determine relative angles/radius of GT groups from UBS.
        ang_grps = (np.random.rand() + np.arange(self.n_grps) / self.n_grps) * 2 * np.pi
        r_min, r_max = 0.2 * self.range_pos, 0.3 * self.range_pos
        r_grps = r_min + np.random.rand(self.n_grps) * (r_max - r_min)
        pos_grps = self.pos_ubs + (np.stack((np.cos(ang_grps), np.sin(ang_grps))) * r_grps).T

        # Randomly sample positions of GTs.
        pos_gts = np.empty((self.n_gts, 2), dtype=np.float32)
        for grp_idx in range(self.n_grps):
            gt_idxes = slice(grp_idx * self.gts_per_grp, (grp_idx + 1) * self.gts_per_grp)
            pos = pos_grps[grp_idx] + 0.25 * self.r_cov * np.random.randn(self.gts_per_grp, 2)
            pos_gts[gt_idxes] = pos

        self.pos_gts = np.clip(pos_gts, 0, self.range_pos)
        np.random.shuffle(self.pos_gts)

    def step(self, action):
        self.t += 1  # One timestep has elapsed.
        move = self.avail_moves[action]  # Get movement of UBS

        self.pos_ubs = np.clip(self.pos_ubs + move, 0, self.range_pos)  # Latest positions of UBSs
        self._transmit_data()  # UBSs provide wireless service for GTs at new positions.

        rew = self._get_reward()  # Get global reward.
        self.ep_ret += rew  # Accumulate episode return.
        done = self._get_terminate()
        info = dict(EpRet=self.ep_ret, EpLen=self.t, AvgGlobalUtility=self.avg_global_util,
                    FairIdx=self.fair_idx, TotalThroughput=self.total_throughput)
        # Mark whether termination of episode is caused by reaching episode limit.
        info['BadMask'] = True if (self.t == self.episode_limit) else False

        if self.recorder is not None:
            self.recorder.click(pos_ubs=self.pos_ubs.copy(), global_utility=self.global_util, reward=rew,
                                total_throughput=self.total_throughput, fair_idx=self.fair_idx,
                                rate_per_gt=self.rate_per_gt.copy(), velocity=np.linalg.norm(move / self.dt))

        return self.get_obs(), rew, done, info

    def _transmit_data(self):
        # Compute distance between entities.
        self.d_u2g = np.zeros(self.n_gts, dtype=np.float32)
        for m in range(self.n_gts):
            self.d_u2g[m] = np.linalg.norm(self.pos_gts[m] - self.pos_ubs)

        # Get interference relations and associate GTs to UBSs.
        self.sched = np.zeros(self.n_gts, dtype=bool)
        for m in self.prior_gts:
            if (self.sched.sum() < self.n_rbs) and (self.d_u2g[m] <= self.r_cov):
                self.sched[m] = True

        g = self.chan.estimate_chan_gain(self.d_u2g, self.h_ubs)  # Channel gain
        p_rx = self.p_tx * g * self.sched  # Rx power of intended signal
        sinr = p_rx / (self.bw * self.n0)  # Signal-to-interference-plus-noise ratio (SINR) of each GT
        self.rate_per_gt = self.bw * np.log2(1 + sinr) * 1e-6  # Achievable rate of each GT (Mbps)

        self.aver_rate_per_gt = (self.aver_rate_per_gt * self.t + self.rate_per_gt) / (self.t + 1)
        self.total_throughput += (self.rate_per_gt.sum() * self.dt / 1e3)  # Total throughput of the network (Gb)
        self.fair_idx = compute_jain_fairness_index(self.aver_rate_per_gt)  # Current fairness index among GTs
        self.global_util = self.fair_idx * self.rate_per_gt.mean()  # Trade-off between fairness and throughput
        self.avg_global_util = (self.avg_global_util * self.t + self.global_util) / (self.t + 1)
        self.prior_gts = np.argsort(self.aver_rate_per_gt)  # Priorities of GTs for next timestep

    def get_obs(self):
        own_feats = np.zeros(self.obs_own_feats_size, dtype=np.float32)
        gt_feats = np.zeros(self.obs_gt_feats_size, dtype=np.float32)

        own_feats[0:2] = self.pos_ubs / self.range_pos

        # GT features
        for m in range(self.n_gts):
            gt_feats[m, 0:2] = (self.pos_gts[m] - self.pos_ubs) / self.range_pos
            gt_feats[m, 2] = self.rate_per_gt[m] / self.max_rate
            gt_feats[m, 3] = self.aver_rate_per_gt[m] / self.max_rate * self.n_grps

        return dict(agent=own_feats, gt=gt_feats)

    def get_obs_size(self):
        return dict(agent=self.obs_own_feats_size, gt=self.obs_gt_feats_size)

    @ property
    def obs_own_feats_size(self):
        nf_own = 2
        return nf_own

    @property
    def obs_gt_feats_size(self):
        nf_gt = 2 + 1 + 1
        return self.n_gts, nf_gt

    def _get_reward(self):
        reward = self.reward_scale_rate * self.global_util / self.max_rate
        return reward

    def _get_terminate(self):
        return True if (self.t == self.episode_limit) else False

    def replay(self, **kwargs):
        if self.recorder is not None:
            self.recorder.replay(**kwargs)


if __name__ == '__main__':
    env = SingleUbsCoverageEnv(n_grps=2, gts_per_grp=2)
    env.reset()
    print(160 / env.max_rate * 2)
    print(env.rate_per_gt)
    # env.replay(show_img=True)
