import os
import os.path as osp
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from algos.drqn.run import load_and_run_policy as test_drqn
from algos.madrqn.run import load_and_run_policy as test_madrqn

from envs import REGISTRY as env_REGISTRY

TEST_FUNCTIONS = {
    'drqn': test_drqn,
    'madrqn': test_madrqn,
}


def insert_data(dataset, exp_name, new_data):
    """Adds new data to dataset."""

    # Each exp_name specifies a configuration. When exp_name is new, add it to dataset.
    if exp_name not in dataset:
        dataset[exp_name] = dict()

    # Results under the same experiment name are merged.
    for k in new_data.keys():
        if k in dataset[exp_name]:
            dataset[exp_name][k] = pd.concat([dataset[exp_name][k], new_data[k]], ignore_index=True)
        else:
            dataset[exp_name][k] = new_data[k]
    return dataset


def test_series(algo_name, metrics, all_logdirs, checkpoint, n_episodes, output_dir):
    """Tests performance of trained agents from a series of experiments."""

    dataset = {}  # Dict holding all test results

    for logdir in all_logdirs:
        # Each logdir holds possibly multiple experiments with the same name/configuration.
        # Their only difference is the random seed, and hence their results are merged together.
        for root, dirs, files in os.walk(logdir):

            if checkpoint in files:
                config_path = open(os.path.join(root, 'config.json'))
                config = json.load(config_path)

                # Get experiment name and random seed.
                exp_name = config['exp_name']
                seed = config['seed']

                # Get environment function and its args.
                env_fn = env_REGISTRY[config['env_fn']]
                env_kwargs = config['env_kwargs']

                # Get the path of saved model parameters and arguments for training.
                model_path = osp.join(root, checkpoint)
                args = list(config['args'].values())[0]

                # Create a subdirectory identified by experiment name to hold test results.
                subdir = osp.join(output_dir, exp_name + f'_seed{seed}')
                os.makedirs(subdir, exist_ok=True)

                # Test performance of the variant.
                test_fn = TEST_FUNCTIONS[algo_name]
                test_rsts = test_fn(model_path, env_fn, env_kwargs, seed, args, n_episodes, subdir)
                dataset = insert_data(dataset, exp_name, test_rsts)

    # Summarize results of each variant into a DataFrame.
    summary = []
    for exp_name in dataset.keys():
        for metric in metrics:
            # summary[metric, exp_name] = dataset[exp_name][metric].to_numpy()
            summary.append(pd.DataFrame(dataset[exp_name][metric].to_numpy(), columns=[np.array([metric]), np.array([exp_name])]))
    summary = pd.concat(summary, axis=1)
    summary.columns.set_names(['metric', 'exp_name'], inplace=True)
    summary = summary.sort_index(axis=1)
    # Write to disk.
    os.makedirs(output_dir, exist_ok=True)
    summary.to_csv(osp.join(output_dir, 'test_summary.csv'))

    # Also create a transposed version.
    cols = pd.MultiIndex.from_product([dataset.keys(), summary.index], names=('exp_name', 'episode'))
    summary_t = pd.DataFrame(columns=cols, index=metrics)
    for metric in metrics:
        for exp_name in dataset.keys():
            vals = summary[metric][exp_name].to_numpy().T
            summary_t.loc[metric, exp_name] = vals
    summary_t.to_csv(osp.join(output_dir, 'test_summary_t.csv'))

    # Plot a box chart for metrics.
    n_rows = 2
    while 2 * n_rows < len(metrics): n_rows += 1
    fig, axes = plt.subplots(nrows=n_rows, ncols=2)
    plt.subplots_adjust(wspace=0.35, hspace=0.5)  # Distance between subplots
    for i, m in enumerate(metrics):
        if m in summary.columns:  # Not all metrics are provided.
            summary[m].plot.box(ax=axes[i // 2, i % 2], figsize=(6, 4))
            axes[i // 2, i % 2].set_title(m)

    plt.savefig(osp.join(output_dir, 'test_summary.png'))
    plt.show()


if __name__ == '__main__':
    base_dir = './data'

    # Test all candidates in experiment 1.
    grps = [2, 3, 4]
    agents = ['rnn', 'gnn']
    metrics = ['EpRet', 'AvgGlobalUtility', 'TotalThroughput', 'FairIdx']
    for n_grps in grps:
        all_logdirs = []
        for agent in agents:
            all_logdirs.append(osp.join(base_dir, f"exp1_grp{n_grps}_{agent}"))
        output_dir = osp.join('./data', 'test_exp1', f'{n_grps}grps')
        test_series('drqn', metrics, all_logdirs, 'checkpoint_epoch50.pt', 10, output_dir)

    # # Test all candidates in experiment 2.
    # maps = ['inf', 'r400', 'r800']
    # agents = ['none', 'none_qmix', 'tarmac', 'disc']
    # metrics = ['EpRet', 'AvgGlobalUtility', 'TotalThroughput', 'FairIdx', 'ProbCollision']
    # for map in maps:
    #     all_logdirs = []
    #     for agent in agents:
    #         all_logdirs.append(osp.join(base_dir, f"exp2_{map}_{agent}"))
    #     output_dir = osp.join('./data', 'test_exp2', map)
    #     test_series('madrqn', metrics, all_logdirs, 'checkpoint_epoch100.pt', 10, output_dir)

    # # Test all candidates in experiment 3.
    # maps = ['4ubs', '6ubs', '8ubs']
    # agents = ['none', 'none_qmix', 'tarmac', 'disc']
    # metrics = ['EpRet', 'AvgGlobalUtility', 'TotalThroughput', 'FairIdx', 'ProbCollision']
    # for map in maps:
    #     all_logdirs = []
    #     for agent in agents:
    #         all_logdirs.append(osp.join(base_dir, f"exp3_{map}_gnn_{agent}"))
    #     output_dir = osp.join('./data', 'test_exp3', map)
    #     test_series('madrqn', metrics, all_logdirs, 'checkpoint_epoch100.pt', 10, output_dir)
