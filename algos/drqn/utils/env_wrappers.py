from abc import abstractmethod

import numpy as np
import torch as th
import dgl

from gym.spaces.utils import flatten_space, flatten


def make_env(env_fn, args):
    env = env_fn()
    return Wrapper(env, args)


class Wrapper(object):
    def __init__(self, env, args):
        self.env = env
        self.agent_type = args.agent
        self.obs_wrapper = FlattenObservation(env) if self.agent_type == 'rnn' else GraphObservation(env)

    def get_obs_size(self):
        return self.obs_wrapper.get_obs_size()

    def get_env_info(self):
        return dict(obs_shape=self.get_obs_size(), n_actions=self.n_actions, episode_limit=self.episode_limit)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)

    def step(self, act):
        obs, rew, done, info = self.env.step(act)
        obs = self.obs_wrapper.observation(obs)
        return obs, rew, done, info

    def reset(self):
        obs = self.env.reset()
        return self.obs_wrapper.observation(obs)


class FlattenObservation:
    def __init__(self, env):
        self.env = env
        self._observation_space = flatten_space(env.observation_space)

    def get_obs_size(self):
        """Assumes that all agents are homogeneous and share the same observation space."""
        return self._observation_space.shape[0]

    def observation(self, obs):
        flattened_obs = flatten(self.env.observation_space, obs)
        return th.as_tensor(flattened_obs, dtype=th.float32).unsqueeze(0)


class GraphObservation:
    def __init__(self, env):
        self.env = env

    def get_obs_size(self):
        return dict(agent=self.env.obs_own_feats_size, gt=self.env.obs_gt_feats_size[1])

    def observation(self, obs):
        n_gts = obs['gt'].shape[0]
        # Define relations between units and agent.
        data_dict = {('gt', 'seen-by', 'agent'): (th.arange(n_gts), th.zeros(n_gts, dtype=th.long))}
        num_nodes_dict = {'gt': n_gts, 'agent': 1}

        # Create heterogeneous graph.
        g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

        # Add node features.
        g.ndata['feat'] = {
            'gt': th.as_tensor(obs['gt']),
            'agent': th.as_tensor(obs['agent']).unsqueeze(0),
        }
        return g


