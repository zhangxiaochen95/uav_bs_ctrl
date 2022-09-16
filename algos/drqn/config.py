DEFAULT_CONFIG = {

    'device': 'cpu',
    'cuda_deterministic': False,

    'agent': 'rnn',  # Agent type

    # Model parameters
    'hidden_size': 256,  # Hidden size of agents
    'n_layers': 2,  # Number of fully-connected layers in rnn agent or flat observation encoder
    'n_heads': 4,  # Number of heads in graph observation encoder

    # Basic training hyperparameters
    'lr': 5e-4,  # Learning rate
    'gamma': 0.99,  # Discount factor
    'polyak': 0.999,  # Interpolation factor in polyak averaging for target network
    'batch_size': 32,  # Minibatch size for SGD
    'replay_size': int(5e4),  # Capacity of replay buffer
    'decay_steps': int(2e5),  # Number of timesteps for exploration
    'max_seq_len': 10,  # Length of data chunks

    'steps_per_epoch': 10000,  # Number of timesteps in each epoch
    'epochs': 50,  # Number of epochs to run
    'update_after': 5000,  # Number of env interactions to collect before starting to do gradient descent updates
    'num_test_episodes': 5,  # Number of episodes in each test
    'save_freq': 10,  # How often (in terms of gap between epochs) to save the current policy and value function

    # Optimization techniques
    'anneal_lr': True,  # Whether lr annealing is used
}
