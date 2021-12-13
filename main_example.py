r"""Main script of training loop"""
import numpy as np
import datetime
import tasks.uav_bs_ctrl as task
from drqn import DRQN, train_drqn
import mods
from utils.logx import makedir

if __name__ == '__main__':
    # Create the environment.
    env = task.UavBaseStationControl(n_steps=200, dt=10, n_grids=16, len_grid=100.,
                                     n_uavs=4, h_uavs=100., vmax_uavs=10, dmin_uavs=0.,
                                     r_cov=150, p_tx=10, n0=-170, bw=180e3, fc=2.4e9, n_chans=10, vmax_gts=0.,
                                     e_types={'uav': False, 'comm': True},
                                     e_thres={'gt': 400, 'uav': np.inf, 'comm': np.inf},
                                     is_fair=True, is_tmnt=False, eps_r=1., eps_p=0.)

    num_train_episodes = 2000  # Number of training episode
    total_steps = num_train_episodes * env.n_steps  # Maximum number of time steps throughout training
    num_tests = 100  # Number of tests

    # Specifies if communication among agents is used and define the neural network (NN) model.
    model = mods.GraphVisCommNet(env.get_obs_size(), env.n_acts, hidden_size={'vis': 64, 'comm': 256}, num_heads=4,
                                 is_het_vis=env.e_types['uav'], is_c_comm=False, num_comm_steps=2)

    # Create agent based on created environment and NN model.
    agent = DRQN(model, env.n_agents, env.n_acts, rnn_dim=model.hidden_size['rnn'], mem_capacity=100000,
                 lr=2.5e-4, wd=1e-2, lr_decay=0.99, gamma=0.99, polyak=0.9995)

    # Specify the directory to save results.
    save_dir = makedir(f'./results/' + model.tag + f'_{datetime.datetime.now().strftime("D%m-%d_T%H-%M")}/')

    # Call the main function to train agent.
    train_drqn(env, agent, batch_size=64, eps_start=1., eps_end=0.1, eps_decay=total_steps / 5,
               update_every=1, num_train_episodes=num_train_episodes,
               test_freq=max(1, int(num_train_episodes / num_tests)), num_test_episodes=20, save_dir=save_dir)
