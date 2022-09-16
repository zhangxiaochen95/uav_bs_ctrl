import copy
from copy import deepcopy
import os
import os.path as osp
from types import SimpleNamespace as SN
from functools import partial
import time

import random
import numpy as np
import pandas as pd
import torch as th

from algos.common import *
from algos.drqn.learner import QLearner
from algos.drqn.utils.env_wrappers import make_env
from algos.drqn.config import DEFAULT_CONFIG
from utils.logx import EpochLogger
from utils.run_utils import setup_logger_kwargs


def train(env_fn, env_kwargs, seed, train_kwargs=dict(), logger_kwargs=dict()):

    # Setup logger.
    logger = EpochLogger(**logger_kwargs)
    del logger_kwargs

    # Set random seeds.
    set_rand_seed(seed)

    # Get configuration.
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update(train_kwargs)
    args = SN(**config)
    del train_kwargs, config
    args = check_args_sanity(args)

    logger.save_config(locals())  # Save configurations of experiment.

    if args.cuda_deterministic:
        # Sacrifice performance to ensure reproducibility.
        # Refer to https://pytorch.org/docs/stable/notes/randomness.html
        th.backends.cudnn.benchmark = False
        th.backends.cudnn.deterministic = True

    env = make_env(partial(env_fn, **env_kwargs, record=False), args)
    test_env = make_env(partial(env_fn, **env_kwargs, record=True), args)

    env_info = env.get_env_info()
    learner = QLearner(env_info, args)

    total_steps = args.steps_per_epoch * args.epochs
    update_after = max(args.update_after, learner.batch_size * learner.max_seq_len)
    update_every = learner.max_seq_len

    eps_start, eps_end = 1, 0.05
    eps_thres = lambda t: max(eps_end, -(eps_start - eps_end) / args.decay_steps * t + eps_start)

    def test_agent():
        """Tests the performance of trained agent."""
        for j in range(args.num_test_episodes):
            o, h, d = test_env.reset(), learner.init_hidden(), False

            while not d:
                a, h = learner.act(o, h, 0.05)  # Take deterministic actions at test time.
                o, _, d, info = test_env.step(a)

            logger.store(TestEpRet=info.get('EpRet'))
            if epoch % args.save_freq == 0:
                test_env.replay(save_dir=osp.join(logger.output_dir, f'epoch{epoch}_episode{j}'))

    episode = 0
    start_time = time.time()
    o, h = env.reset(), learner.init_hidden()

    for t in range(total_steps):
        # Select actions following epsilon-greedy strategy.
        a, h2 = learner.act(o, h, eps_thres(t))
        # Call environment step.
        o2, r, d, info = env.step(a)
        # Store experience to replay buffer.
        learner.cache(o, h, a, r, o2, h2, d, info.get('BadMask'))
        # Move to next timestep.
        o, h = o2, h2
        # Reach the end of an episode.
        if d:
            episode += 1
            logger.store(**info)
            o, h = env.reset(), learner.init_hidden()  # Reset env and RNN hidden states.

        # Update parameters of model.
        if (t >= update_after) and (t % update_every == 0):
            diagnostic = learner.update()
            logger.store(**diagnostic)

        # End of epoch handling
        if (t + 1) % args.steps_per_epoch == 0:
            # Test performance of trained agents.
            epoch = (t + 1) // args.steps_per_epoch
            # Test performance of trained agents.
            test_agent()
            # Call learning rate annealing.
            if learner.anneal_lr:
                learner.lr_scheduler.step()
            # Save model parameters.
            if (epoch % args.save_freq == 0) or (epoch == args.epochs):
                save_path = osp.join(logger.output_dir, f'checkpoint_epoch{epoch}.pt')
                learner.save_checkpoint(save_path, stamp=dict(epoch=epoch, t=t))

            # Log info about epoch.
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('Episode', episode)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('AvgGlobalUtility', with_min_and_max=True)
            logger.log_tabular('FairIdx', average_only=True)
            logger.log_tabular('TotalThroughput', average_only=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('TotalEnvInteracts', t + 1)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('ExploreEps', eps_thres(t))
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()

    print("Complete.")


def load_and_run_policy(model_path, env_fn, env_kwargs, seed, agent_kwargs, n_episodes, output_dir):
    """Loads stored model parameters and runs learnt policy."""

    # Set random seeds.
    set_rand_seed(seed)

    # Setup configuration.
    config = deepcopy(DEFAULT_CONFIG)
    config.update(agent_kwargs)

    args = SN(**config)
    args = check_args_sanity(args)

    if args.cuda_deterministic:
        # Sacrifice performance to ensure reproducibility.
        # Refer to https://pytorch.org/docs/stable/notes/randomness.html
        th.backends.cudnn.benchmark = False
        th.backends.cudnn.deterministic = True

    # Instantiate env.
    env = make_env(partial(env_fn, **env_kwargs, record=True), args)

    # Create agent and load parameters.
    env_info = env.get_env_info()
    learner = QLearner(env_info, args)
    learner.load_checkpoint(model_path)

    # Run multiple test episodes and record performance of trained agent in each episode.
    rsts = {}  # Value of each item is a list holding metrics specified by key across episodes.
    for n in range(n_episodes):
        o, h, d = env.reset(), learner.init_hidden(), False  # Reset env and RNN hidden states.
        while not d:
            a, h = learner.act(o, h, 0.05)  # Take (quasi) deterministic actions at test time.
            o, _, d, info = env.step(a)  # Env step

        # Call replay function of the recorder.
        env.replay(save_dir=osp.join(output_dir, f'episode{n}'))
        # Collect metrics evaluate performance through the episode.
        for k, v in info.items():
            if k not in rsts:
                rsts[k] = []
            rsts[k].append(v)

    # Create a DataFrame to hold metrics across episodes.
    rsts = pd.DataFrame(rsts)
    return rsts


if __name__ == '__main__':
    import argparse
    from envs.subs_cov.subs_cov import SingleUbsCoverageEnv

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--exp', type=str, default='drqn')
    args = parser.parse_args()

    logger_kwargs = setup_logger_kwargs(args.exp, args.seed)
    train_kwargs = dict(agent='rnn')

    train(SingleUbsCoverageEnv, dict(), args.seed, train_kwargs=train_kwargs, logger_kwargs=logger_kwargs)