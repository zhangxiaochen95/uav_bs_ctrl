import os
import os.path as osp
import json
import pandas as pd


def collect_curves(all_logdirs, xaxis, metric, output_dir):
    """
    Goes through a list of experiment directories,
    extract x_label vs. metric curve in each experiment
    and saves all curves to a .csv file.
    """
    dataset = []
    for logdir in all_logdirs:
        for root, dirs, files in os.walk(logdir):
            if 'progress.txt' in files:
                try:
                    config_path = open(os.path.join(root, 'config.json'))
                    config = json.load(config_path)

                    exp_name = config['exp_name']
                    seed = config['seed']
                    exp_data = pd.read_table(os.path.join(root, 'progress.txt'))

                    identifier = pd.MultiIndex.from_tuples([(exp_name, f'seed{seed}')])  # Identifier of experiment
                    index = exp_data[xaxis]  # x
                    if xaxis == 'TotalEnvInteracts':
                        index /= 1e6
                    exp_data = pd.DataFrame(exp_data[metric].to_numpy(), index=index, columns=identifier)
                    dataset.append(exp_data)

                except:
                    print('Could not read from %s' % root)
                    continue

    # Concatenate all curves in dataset.
    dataset = pd.concat(dataset, axis=1)
    # Write dataset to disk.
    os.makedirs(output_dir, exist_ok=True)
    dataset.to_csv(osp.join(output_dir, f'{xaxis}_vs_{metric}.csv'))


if __name__ == '__main__':
    base_dir = './data'

    # Collect training curves in experiment 1.
    grps = [2, 3, 4]
    agents = ['rnn', 'gnn']
    all_logdirs = []
    for grp in grps:
        for agent in agents:
            all_logdirs.append(osp.join(base_dir, f"exp1_grp{grp}_{agent}"))
    collect_curves(all_logdirs, 'TotalEnvInteracts', 'AverageEpRet', './data/exp1_curves')

    # Collect training curves in experiment 2.
    maps = ['r400', 'r800', 'inf']
    agents = ['none', 'none_qmix', 'tarmac', 'disc']
    all_logdirs = []
    for map in maps:
        for agent in agents:
            all_logdirs.append(osp.join(base_dir, f"exp2_{map}_{agent}"))
    collect_curves(all_logdirs, 'TotalEnvInteracts', 'AverageEpRet', './data/exp2_curves')

    # Collect training curves of GNN agents in experiment 3.
    maps = ['4ubs', '6ubs', '8ubs']
    agents = ['none', 'none_qmix', 'tarmac', 'disc']

    all_logdirs = []
    for map in maps:
        for agent in agents:
            all_logdirs.append(osp.join(base_dir, f"exp3_{map}_gnn_{agent}"))
    collect_curves(all_logdirs, 'TotalEnvInteracts', 'AverageEpRet', './data/exp3_curves')

    # Collect training curves of MLP agents in experiment 3.
    maps = ['4ubs']
    agents = ['none', 'none_qmix', 'tarmac', 'disc']

    all_logdirs = []
    for map in maps:
        for agent in agents:
            all_logdirs.append(osp.join(base_dir, f"exp3_{map}_mlp_{agent}"))
    collect_curves(all_logdirs, 'TotalEnvInteracts', 'AverageEpRet', './data/exp3_mlp_curves')
