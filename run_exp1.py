r"""Run a series of experiments with different hyperparameters."""
from utils.run_utils import ExperimentGrid

from algos.drqn.run import train as drqn

ALGOS = {
    'drqn': drqn,
}

if __name__ == '__main__':
    num_runs = 3
    run_kwargs = {'num_cpu': 1, 'data_dir': None, 'datestamp': False}

    from envs.subs_cov.subs_cov import SingleUbsCoverageEnv

    algo_name = 'drqn'
    eg = ExperimentGrid(name='exp1')
    eg.add('seed', [10 * (i + 1) for i in range(num_runs)])  # Assign different seeds for multiple runs of experiments.

    eg.add('env_fn', SingleUbsCoverageEnv,)
    eg.add('env_kwargs:n_grps', [2, 3, 4], 'grp')
    eg.add('env_kwargs:gts_per_grp', [5], 'size')

    # Set device.
    eg.add('train_kwargs:device', 'cpu',)
    eg.add('train_kwargs:cuda_index', 0,)

    # Setup model architecture.
    eg.add('train_kwargs:agent', ['rnn', 'gnn'], '',)

    # Set training hyperparameters.
    eg.add('train_kwargs:lr', 5e-4, 'lr')
    eg.add('train_kwargs:polyak', 0.999, 'polyak')
    eg.add('train_kwargs:replay_size', int(5e4), 'mem')
    eg.add('train_kwargs:decay_steps', int(2e5), 'dec')

    eg.add('train_kwargs:epochs', 50)
    eg.add('train_kwargs:steps_per_epoch', 20000)
    eg.add('train_kwargs:update_after', 10000)

    eg.run(ALGOS[algo_name], **run_kwargs)
