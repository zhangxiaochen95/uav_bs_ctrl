from abc import abstractmethod

import numpy as np
import torch
import dgl

from gym.spaces.utils import flatten_space, flatten
from envs.multi_agent_env import MultiAgentWrapper
from algos.madrqn.utils.reward_normalizer import ZFilter


def make_env(env_fn, args):
    """Instantiates and wraps env."""
    env = env_fn()
    return MultiUbsCoverageWrapper(env, args)


# ==================================================================================================================== #
# Wrapper of local observations


class LocalObservationWrapper:
    """Base class of local observation wrappers"""
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)

    @ abstractmethod
    def get_obs_size(self):
        raise NotImplementedError

    @abstractmethod
    def local_observation(self, local_obs):
        raise NotImplementedError


class FlattenedObservation(LocalObservationWrapper):
    """Transforms dict observations to flattened observations (1-D ndarray)"""

    def __init__(self, env):
        super(FlattenedObservation, self).__init__(env)
        self._observation_space = flatten_space(self.env.observation_space[0])

    def get_obs_size(self):
        return self._observation_space.shape[0]

    def local_observation(self, local_obs):
        flat_local_obs = [flatten(self.env.observation_space[0], o) for o in local_obs]
        return torch.as_tensor(np.stack(flat_local_obs), dtype=torch.float32)


class GraphObservation(LocalObservationWrapper):
    """Transforms dict observations to a batched heterogeneous graph (DGLGraph)"""

    def __init__(self, env):
        super(GraphObservation, self).__init__(env)

    def get_obs_size(self):
        return dict(agent=self.obs_own_feats_size, ubs=self.obs_ubs_feats_size[1] - 1, gt=self.obs_gt_feats_size[1] - 1)

    def local_observation(self, local_obs):
        local_obs_graphs = [self.build_obs_graph(o) for o in local_obs]
        return dgl.batch(local_obs_graphs)

    def build_obs_graph(self, obs):
        # Get the IDs of GTs and UBSs nearby agent.
        gt_ids, ubs_ids = np.equal(obs['gt'][:, 0], 1), np.equal(obs['ubs'][:, 0], 1)
        num_gts, num_ubs = gt_ids.sum(), ubs_ids.sum()

        # Define relations between units and agent.
        data_dict = {
            ('gt', 'seen', 'agent'): (torch.arange(num_gts), torch.zeros(num_gts, dtype=torch.long)),
            ('ubs', 'near', 'agent'): (torch.arange(num_ubs), torch.zeros(num_ubs, dtype=torch.long)),
            ('agent', 'talk', 'agent'): ([], []),
        }
        num_nodes_dict = {'gt': num_gts, 'ubs': num_ubs, 'agent': 1}
        g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

        # Add node features.
        g.ndata['feat'] = {
            'gt': torch.as_tensor(obs['gt'][gt_ids, 1:]),
            'ubs': torch.as_tensor(obs['ubs'][ubs_ids, 1:]),
            'agent': torch.as_tensor(obs['agent']).unsqueeze(0),
        }
        return g

# ==================================================================================================================== #
# Organized wrapper for `MultiUbsCoverageEnv`


class MultiUbsCoverageWrapper(MultiAgentWrapper):
    def __init__(self, env, args):
        super(MultiUbsCoverageWrapper, self).__init__(env)

        self._enc_type = args.o  # Type of observation encoder
        self._comm_protocol = args.c  # Multi-agent communication protocol

        # Choose local observation wrapper based on encoder type.
        if self._enc_type == 'mlp':
            self.local_obs_wrapper = FlattenedObservation(env)
        elif self._enc_type == 'gnn':
            self.local_obs_wrapper = GraphObservation(env)

        # Create reward normalizer.
        self._normalize_reward = args.norm_r
        if self._normalize_reward:
            reward_shape = 1 if args.share_reward else self.n_agents
            self.reward_normalizer = ZFilter(shape=(1, reward_shape), clip=10)

    def get_env_info(self):
        """Returns required info by learners."""
        return dict(obs_shape=self.get_obs_size(), state_shape=self.get_state_size(), n_actions=self.n_actions,
                    n_agents=self.n_agents, episode_limit=self.episode_limit)

    def get_obs_size(self):
        return self.local_obs_wrapper.get_obs_size()

    def observation(self, obs):
        """Wraps list of local observations into an object."""

        # Wrap local observations to meet requirements of encoder.
        local_obs = self.local_obs_wrapper.local_observation(obs)

        # When MAC is used, put local observations into a communication graph.
        if self._comm_protocol is None:
            return local_obs
        else:
            comm_graph = self.build_comm_graph()
            if isinstance(self.local_obs_wrapper, FlattenedObservation):
                comm_graph.nodes['agent'].data['feat'] = local_obs
                return comm_graph
            else:
                return dgl.merge([local_obs, comm_graph])

    def build_comm_graph(self):
        u, v = [], []
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if self.d_u2u[i, j] <= self.r_comm:
                    u.append(i), v.append(j)

        g = dgl.heterograph(
            {
                ('gt', 'seen', 'agent'): ([], []),
                ('ubs', 'near', 'agent'): ([], []),
                ('agent', 'talk', 'agent'): (u, v),
            },
            num_nodes_dict={'gt': 0, 'ubs': 0, 'agent': self.n_agents}
        )
        return g

    def state(self, state):
        """Converts state dict into flattened ndarray"""
        return torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)

    def reward(self, reward):
        if self._normalize_reward:
            reward = self.reward_normalizer(reward)
        return reward

