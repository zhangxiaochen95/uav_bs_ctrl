DEFAULT_CONFIG = {

    'device': 'cpu',
    'cuda_deterministic': False,
    'cuda_index': 0,

    'o': 'mlp',  # Type of observation encoder
    'c': None,  # Protocol for multi-agent communication
    'share_reward': False,

    # Model parameters
    'hidden_size': 64,  # Hidden size of agents
    'n_layers': 1,  # Number of fully-connected layers in rnn agent or flat observation encoder
    'n_heads': 4,  # Number of attention heads in graph observation encoder
    'msg_size': 64,  # Size of messages
    'key_size': 16,  # Size of signature and queries in TarMAC
    'n_rounds': 1,  # Number of communication rounds if multi-round communication is supported
    'embed_dim': 32,  # Dimension of Mixer

    # Basic training hyperparameters
    'lr': 5e-4,  # Learning rate
    'gamma': 0.99,  # Discount factor
    'polyak': 0.995,  # Interpolation factor in polyak averaging for target network
    'batch_size': 32,  # Minibatch size for SGD
    'replay_size': int(5e3),  # Capacity of replay buffer
    'decay_steps': int(5e4),  # Number of timesteps for exploration
    'max_seq_len': None,

    'steps_per_epoch': 4000,  # Number of timesteps in each epoch
    'epochs': 50,  # Number of epochs to run
    'update_after': 2000,  # Number of env interactions to collect before starting to do gradient descent updates
    'num_test_episodes': 5,  # Number of episodes in each test
    'save_freq': 10,  # How often (in terms of gap between epochs) to save the current policy and value function

    # Optimization techniques
    'anneal_lr': True,  # Whether lr annealing is used
    'norm_r': True,  # Whether reward normalization is used
    'double_q': False,  # Whether double Q-learning is used
    'dueling': False,  # Whether dueling architecture is used
    'mixer': False,  # Whether QMix is used
}
