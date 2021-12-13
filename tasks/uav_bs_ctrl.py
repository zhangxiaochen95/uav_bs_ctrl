r"""Environments which agents interact with."""
from copy import deepcopy
from random import sample
import numpy as np
import torch
import dgl
import tasks.physics as physics
import tasks.init_fn as init_fn


class UavBaseStationControl(object):
    """Flight control of multi-UAV-BSs for throughput maximization of GTs"""
    def __init__(self, n_steps=200, dt=10, n_grids=10, len_grid=100.,
                 n_uavs=4, h_uavs=100., vmax_uavs=10, dmin_uavs=0.,
                 r_cov=100, p_tx=10, n0=-170, bw=180e3, fc=2.4e9, n_chans=5, vmax_gts=0.,
                 e_types={'uav': True, 'comm': True}, e_thres={'gt': 400, 'uav': 600, 'comm': np.inf},
                 is_fair=True, is_tmnt=False, eps_r=1., eps_p=0.):
        self.n_steps = n_steps   # Maximum number of time steps in each episode
        self.dt = dt  # Length of each time step

        self.n_grids = n_grids  # Number of grids in each direction
        self.len_grid = len_grid  # Length of grids (m)
        self.len_area = n_grids * len_grid  # Length of area (m)

        # Basic features of UAV-BSs and GTs
        self.n_uavs = n_uavs  # Number of UAV-BSs
        self.h_uavs = h_uavs  # Flying height of UAV-BSs (m)
        self.vmax_uavs = self.dt * vmax_uavs  # Maximum velocity of UAV-BSs (m/step)
        self.dmin_uavs = dmin_uavs  # Minimum safe distance between UAV-BSs (m)
        self.p_uavs = None  # Initial positions of UAV-BSs

        self.vmax_gts = vmax_gts  # Maximum velocity of GTs (m/step)
        self.n_gts, self.p_gts = None, None  # Number and positions of GTs

        # Parameters of communication coverage, sensing and cooperation relations
        self.r_cov = r_cov  # Horizontal range of service coverage (m)
        self.p_tx = physics.dbm2w(p_tx)  # Transmit power (Watt)
        self.n0 = physics.dbm2w(n0)  # PSD of AWGN (Watt/Hz)
        self.bw = bw  # Bandwidth of each channel (Hz)
        self.fc = fc  # Carrier frequency (Hz)
        self.n_chans = n_chans  # Number of available sub-channels
        self.chan_model = physics.AirToGroundChannel(self.fc, 'urban')  # Channel model

        self.e_types = e_types  # Edge types
        self.e_thres = e_thres  # Threshold to define edges

        # Parameters related to MARL
        self._is_fair = is_fair  # Whether fairness is incorporated in reward signal
        self._is_tmnt = is_tmnt  # Whether the episode terminate when accident occurs
        self.eps_r, self.eps_p = eps_r, eps_p  # Coefficients of reward and penalty

        self.n_agents = self.n_uavs  # Number of agents
        self.avail_acts = np.array([[-1., 0], [1, 0], [0, -1], [0, 1], [0, 0]])  # Possible actions of each UAV-BS
        self.n_acts = self.avail_acts.shape[0]  # Number of actions
        self.idle_act = torch.tensor(self.n_agents * [self.n_acts - 1], dtype=torch.long).view(self.n_agents, 1)  # Actions letting agents do nothing
        print(f"env_info: \n{self.get_env_info()}")

        # Other parameters
        self.t = None  # Number of elapsed time steps
        self.dist = {'gt': None, 'uav': None}  # Horizontal distance between entities
        self.adj = {'gt': None, 'uav': None, 'comm': None}  # Adjacency matrix for different relations
        self.sch = None  # Scheduling relation between GTs and UAV-BSs
        self.cov_score = None  # Coverage score until now
        self.prior_gts = None  # Priority of GTs for receiving service

        self.rate = None  # Instant rate of each GTs
        self.avg_rate = None  # Average rate until now
        self.fair_idx = None  # Jain's fairness index
        self.r_glo = None  # Global reward signal
        self.ind_p = np.empty(self.n_uavs, dtype=np.bool8)  # Indicators of penalty on each UAV-BS

    def reset(self):
        """Resets the environment. Call `env.reset()` at the beginning of each episode."""
        self.t = 0
        self.p_uavs = init_fn.setup_uavs(self.n_grids, self.len_grid, self.n_uavs)
        self.n_gts, self.p_gts = init_fn.setup_gts(self.n_grids, self.len_grid, 0.1)

        self.cov_score, self.avg_rate = np.zeros(self.n_gts), np.zeros(self.n_gts)
        self.prior_gts = sample(range(self.n_gts), self.n_gts)
        self._update(self.idle_act)
        # Randomness of initial priority is damaged after calling `_update`.
        self.prior_gts = sample(range(self.n_gts), self.n_gts)  # Shuffle the priority.

    def _update(self, act):
        """Updates related parameters according to actions of agents."""
        dpos_uavs = self.vmax_uavs * self.avail_acts[act.to('cpu')].reshape(self.n_uavs, 2)  # Displacement of UAV-BSs
        self.p_uavs += dpos_uavs
        self.v_uavs = np.sqrt(np.square(dpos_uavs).sum(1))  # Instant velocity of UAV-BSs

        # Update positions of GTs.
        if self.vmax_gts > 0:
            ang = 2 * np.pi * np.random.rand(self.n_gts, 1)  # Moving angle
            dist = self.vmax_gts * np.random.rand(self.n_gts, 1)  # Moving distance
            dpos_gts = dist * np.concatenate((np.cos(ang), np.sin(ang)), 1)  # Displacement of GTs
            self.pos_gts = np.clip(self.pos_gts + dpos_gts, 0, self.len_area)  # Restrict p_gts to area.

        def get_distance():
            """Computes the distance between GTs and UAV-BSs based on their positions."""
            dist = {'gt': np.empty((self.n_gts, self.n_uavs)),
                    'uav': np.empty((self.n_uavs, self.n_uavs))}
            for m in range(self.n_uavs):
                for i in range(self.n_gts):
                    dist['gt'][i, m] = np.linalg.norm(self.p_gts[i] - self.p_uavs[m], 2)
                for n in range(self.n_uavs):
                    dist['uav'][n, m] = np.linalg.norm(self.p_uavs[n] - self.p_uavs[m], 2)
            return dist

        def get_penalty():
            """Determines the penalty imposed on each UAV-BS."""
            ind_p = np.zeros(self.n_uavs, dtype=np.bool8)
            for m in range(self.n_uavs):
                # Impose penalty due to collision.
                for n in range(self.n_uavs):
                    if (m != n) and (self.dist['uav'][m, n] < self.dmin_uavs) and (self.v_uavs[m] != 0):
                        ind_p[m] = 1
                # Impose penalty due to boundary violation.
                if (self.p_uavs[m] < 0).any() or (self.p_uavs[m] > self.len_area).any():
                    ind_p[m] = 1
            return ind_p

        # Update distance.
        self.dist = get_distance()
        # Get indicators of penalty.
        self.ind_p = get_penalty()
        if self.ind_p.any():
            # Cancel the movement of penalized UAV-BSs.
            self.p_uavs -= np.expand_dims(self.ind_p, -1) * dpos_uavs
            # Refresh distance if movement of any UAV-BS is cancelled.
            self.dist = get_distance()

        def get_adjacency():
            """Gets adjacency matrices for different relations."""
            return {'gt': (self.dist['gt'] <= self.e_thres['gt']),
                    'uav': (self.dist['uav'] <= self.e_thres['uav']) if self.e_types['uav'] else None,
                    'comm': (self.dist['uav'] <= self.e_thres['comm']) if self.e_types['comm'] else None}

        def get_schedule():
            """Gets the scheduling between GTs and UAV-BSs."""
            sch = np.zeros((self.n_gts, self.n_uavs), dtype=np.bool8)
            for i in self.prior_gts:
                # print(f"GT-{i}'s turn to select.")
                uav_choices = np.argsort(self.dist['gt'][i])
                for m in uav_choices:
                    if (sch[:, m].sum() < self.n_chans) and (self.dist['gt'][i, m] <= self.r_cov):
                        sch[i, m] = 1
                        break
            return sch

        def get_rate():
            """Computes the achievable rate of each GT."""
            chan_gain = self.chan_model.compute_chan_gain(self.dist['gt'], self.h_uavs)  # Channel gain matrix
            inf_rel = (self.dist['gt'] <= self.r_cov)  # Interference relations
            p_rx = self.p_tx * chan_gain * self.sch  # Rx power of useful signal
            p_inf = self.p_tx * chan_gain * inf_rel - p_rx  # Interference level
            sinr = p_rx.sum(1) / (p_inf.sum(1) + self.bw * self.n0)  # Signal-to-interference-plus-noise ratio
            rate = self.bw * np.log2(1 + sinr) * 1e-6  # Rate of each GT (Mbps)
            return rate

        self.adj = get_adjacency()
        self.sch = get_schedule()
        self.rate = get_rate()
        # print(f"currently, self.sch = \n{self.sch}")

        self.tot_tp = self.dt * self.rate.sum()  # Total throughput
        self.fair_idx = physics.jain_fairness_index(self.avg_rate)  # Fairness index
        self.cov_score = (self.t * self.cov_score + self.sch.sum(1)) / (self.t + 1)  # Coverage score
        self.avg_rate = (self.t * self.avg_rate + self.rate) / (self.t + 1)  # Average rate
        self.r_glo = self.eps_r * self.fair_idx * self.rate.sum() if self._is_fair else self.eps_r * self.rate.sum()  # Global reward

        self.prior_gts = np.argsort(self.avg_rate)
        # print(f"Then, prior_gts is updated as {self.prior_gts}")

    def get_obs(self):
        """Gets observations of each agents."""

        def get_feat():
            """Extract useful features of environment."""
            # Features of UAV-BSs
            rel_uav_feats, rel_gt_feats = [], []
            for m in range(self.n_uavs):
                # rel_uav_feats.append(np.concatenate(((self.pos_uavs - self.pos_uavs[m]) / self.len_area,
                #                                      (self.srv_rel.sum(0).reshape(-1, 1) == self.n_chans)), axis=1))
                rel_uav_feats.append((self.p_uavs - self.p_uavs[m]) / self.len_area)
                rel_uav_feats[m][m, 0:2] = self.p_uavs[m] / self.len_area
                if not self._is_fair:
                    rel_gt_feats.append(np.concatenate(((self.p_gts - self.p_uavs[m]) / self.len_area,
                                                        self.sch.sum(1, keepdims=True)), axis=1))
                else:
                    rel_gt_feats.append(np.concatenate(((self.p_gts - self.p_uavs[m]) / self.len_area,
                                                        self.cov_score.reshape(self.n_gts, 1),
                                                        self.sch.sum(1, keepdims=True),
                                                        self.sch[:, m].reshape(self.n_gts, 1)), axis=1))
            return {'uav': rel_uav_feats, 'gt': rel_gt_feats}

        def build_graph(feat):
            """
            Builds the hetergeneous graph from env features.
            First, local observation graph at each UAV-BS is built.
            If communication between UAV-BSs is permitted, local observations are connected with `comm` edges.
            """

            uav_feat, gt_feat = feat['uav'], feat['gt']
            local_obs = []
            for m in range(self.n_uavs):
                local_uav_feat = uav_feat[m] if isinstance(uav_feat, list) else uav_feat
                local_gt_feat = gt_feat[m] if isinstance(gt_feat, list) else gt_feat

                n_local_gts = self.adj['gt'][:, m].sum()
                masked_gt_feat = local_gt_feat[self.adj['gt'][:, m]]

                data_dict = {('gt', 'found-by', 'agent'): (torch.arange(n_local_gts, dtype=torch.long),
                                                           torch.zeros(n_local_gts, dtype=torch.long)),
                             ('agent', 'talks-to', 'agent'): (torch.arange(0), torch.arange(0))}
                feat_dict = {'gt': torch.as_tensor(masked_gt_feat, dtype=torch.float),
                             'agent': torch.as_tensor(local_uav_feat[m].reshape(1, -1), dtype=torch.float)}
                num_nodes_dict = {'gt': n_local_gts, 'agent': 1}

                if self.e_types['uav']:
                    mask_uavs = deepcopy(self.adj['uav'][:, m])
                    mask_uavs[m] = False
                    n_local_uavs = mask_uavs.sum()
                    masked_uav_feats = local_uav_feat[mask_uavs]

                    data_dict[('uav', 'close-to', 'agent')] = (torch.arange(n_local_uavs, dtype=torch.long),
                                                               torch.zeros(n_local_uavs, dtype=torch.long))
                    feat_dict['uav'] = torch.as_tensor(masked_uav_feats, dtype=torch.float)
                    num_nodes_dict['uav'] = n_local_uavs

                vg = dgl.heterograph(data_dict=data_dict, num_nodes_dict=num_nodes_dict)
                vg.ndata['feat'] = feat_dict
                local_obs.append(vg)
            if not self.e_types['comm']:
                return local_obs
            else:
                hg = dgl.batch(local_obs)
                # Add communication channels between agents.
                src_comm, dst_comm = [], []
                for m in range(self.n_uavs):
                    for n in range(self.n_uavs):
                        if (m != n) and self.adj['comm'][m, n]:
                        # if comm_rel[m, n]:
                            src_comm.append(m)
                            dst_comm.append(n)
                hg.add_edges(src_comm, dst_comm, etype=('agent', 'talks-to', 'agent'))
                return hg
        feat = get_feat()
        obs = build_graph(feat)
        return obs

    def _get_reward(self):
        """Gets the reward signal of each UAV-BS."""
        return self.r_glo * np.ones(self.n_uavs) - self.eps_p * self.ind_p

    def _get_terminate(self):
        """Decides if the episode terminates."""
        return self._is_tmnt * self.ind_p

    def step(self, act):
        """Carries out environment transition triggered by actions of agents."""
        self._update(act)
        # print(f"Inside, p_uavs = {self.p_uavs}")
        next_obs = self.get_obs()  # Get the latest observation.
        rwd = self._get_reward()  # Get reward signal.
        done = self._get_terminate()  # Get termination signal
        info = {'tot_tp': self.tot_tp, 'fair_idx': self.fair_idx, 'r_glo': self.r_glo,
                'tot_p': self.eps_p * self.ind_p.sum(), 'n_cov': self.sch.sum()}

        self.t += 1  # One time step has elapsed.
        return next_obs, rwd, done, info

    def get_env_info(self):
        """Returns all attributes of the environment as a dict."""
        return self.__dict__

    def get_obs_size(self):
        """Gets the size of entries in observation."""
        size_uav_feat = 2
        size_gt_feat = 5 if self._is_fair else 3
        return {'gt': size_gt_feat, 'uav': size_uav_feat, 'agent': size_uav_feat}
