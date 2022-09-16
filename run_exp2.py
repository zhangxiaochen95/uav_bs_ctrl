r"""Run a grids of experiments with different hyperparameters."""
from utils.run_utils import ExperimentGrid
from algos.madrqn.run import train as madrqn

ALGOS = {
    'madrqn': madrqn,
}

if __name__ == '__main__':
    from envs.mubs_cov.mubs_cov import MultiUbsCoverageEnv

    algo_name = 'madrqn'
    num_runs = 3
    run_kwargs = {'num_cpu': 1, 'data_dir': None, 'datestamp': False}

    eg = ExperimentGrid(name='exp2')
    eg.add('seed', [10*i for i in range(num_runs)])  # Assign different seeds for multiple runs of experiments.

    # Define env specs.
    eg.add('env_fn', MultiUbsCoverageEnv,)
    eg.add('env_kwargs:map_id', ['inf', 'r400', 'r800'], '', True)
    eg.add('env_kwargs:fair_service', True, 'fair')
    eg.add('env_kwargs:avoid_collision', True, 'collide')

    # Set device.
    eg.add('train_kwargs:device', 'cuda',)
    eg.add('train_kwargs:cuda_index', 0,)

    # Set observation encoder and MAC protocol.
    eg.add('train_kwargs:o', 'mlp', '')
    eg.add('train_kwargs:c', [None, 'tarmac', 'disc'], '', True)

    # Setup model architecture.
    eg.add('train_kwargs:hidden_size', 256, 'hid')
    eg.add('train_kwargs:n_layers', 2, 'l')
    eg.add('train_kwargs:msg_size', 64, 'msg')

    # Set training hyperparameters.
    eg.add('train_kwargs:lr', [2.5e-4], 'lr')
    eg.add('train_kwargs:polyak', 0.999, 'polyak')
    eg.add('train_kwargs:decay_steps', int(5e4), 'dec')
    eg.add('train_kwargs:replay_size', int(5e3), 'mem')
    eg.add('train_kwargs:max_seq_len', None, 'seq')

    eg.add('train_kwargs:epochs', 100)
    eg.add('train_kwargs:steps_per_epoch', 20000)
    eg.add('train_kwargs:update_after', 10000)

    eg.add('train_kwargs:norm_r', True, 'normr')
    eg.add('train_kwargs:anneal_lr', True, '')
    eg.add('train_kwargs:mixer', False, 'qmix', True)  # Set to True when using QMIX
    eg.add('train_kwargs:double_q', True, 'double_q')
    eg.add('train_kwargs:dueling', False, 'duel')

    eg.run(ALGOS[algo_name], **run_kwargs)
