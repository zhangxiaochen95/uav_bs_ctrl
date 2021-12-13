from copy import deepcopy
import time
import datetime
import math
import random
from collections import deque, namedtuple
import numpy as np
import torch
import torch.nn as nn
import dgl
from utils.logx import makedir, MetricLogger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # if gpu is to be used
Transition = namedtuple('Transition', ('obs', 'z', 'act', 'next_obs', 'next_z', 'rwd', 'done'))


class ReplayMemory(object):
    """Replay memory storing experiences of agent"""

    def __init__(self, capacity, entry_keys=Transition):
        self.memory = deque([], maxlen=capacity)
        self.entry_keys = entry_keys

    def push(self, *args):
        """Saves a transition"""
        self.memory.append(self.entry_keys(*args))

    def sample(self, batch_size):
        """Selects a random batch of transitions."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DRQN(nn.Module):
    def __init__(self, model, n_agents, n_acts, rnn_dim, mem_capacity, lr, wd, lr_decay, gamma, polyak):
        super(DRQN, self).__init__()

        self.n_agents = n_agents  # Number of agent
        self.n_acts = n_acts  # Number of actions
        self.rnn_dim = rnn_dim  # Dimension of RNN hidden states

        self.policy_net = model.to(device)  # Neural network to be optimized.
        self.target_net = deepcopy(self.policy_net)  # Copy of policy net as target.
        self.target_net.eval()  # Set target network to eval mode.

        self.memory = ReplayMemory(mem_capacity)  # Replay memory
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=lr, weight_decay=wd)  # AdamW optimizer
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=lr_decay)  # lr scheduler

        self.gamma = gamma  # Discount factor
        self.polyak = polyak  # Interpolation factor in polyak averaging for target networks

        print(f"agent_info:\n{self.__dict__}")

    def select_action(self, obs, z, eps_thres):
        """Each agent selects an action following an epsilon-greedy policy"""
        if isinstance(obs, list):
            obs = dgl.batch(obs)
        if random.random() > eps_thres:  # Exploitation (Greedy action)
            with torch.no_grad():
                logits, next_z = self.policy_net(obs, z)
                act = logits.max(1)[1].view(-1, 1)
        else:  # Exploration (Random action)
            act = torch.randint(0, self.n_acts, size=(self.n_agents, 1), device=device, dtype=torch.long)
            next_z = torch.zeros(self.n_agents, self.rnn_dim, device=device)
        return act, next_z

    def cache(self, obs, z, act, next_obs, next_z, rwd, done):
        """Stores experience of all agents into the replay memory."""
        if not isinstance(obs, list):
            self.memory.push(obs, z, act, next_obs, next_z,
                             torch.tensor(rwd, device=device, dtype=torch.float),
                             torch.tensor(done, device=device, dtype=torch.float))
        # For independent agents, entries of each agent are saved as independent experience.
        else:
            for m in range(self.n_agents):
                self.memory.push(obs[m], z[m].unsqueeze(0), act[m].view(1, 1), next_obs[m], next_z[m].unsqueeze(0),
                                 torch.tensor([rwd[m]], device=device, dtype=torch.float),
                                 torch.tensor([done[m]], device=device, dtype=torch.float))

    def optimize(self, batch_size):
        """Updates the weights of policy network."""
        transitions = self.memory.sample(batch_size)
        # Convert batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        o = dgl.batch(batch.obs)
        next_o = dgl.batch(batch.next_obs)
        z = torch.cat(batch.z)
        next_z = torch.cat(batch.next_z)
        a = torch.cat(batch.act)
        r = torch.cat(batch.rwd)
        d = torch.cat(batch.done)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken.
        logits, z_p = self.policy_net(o, z)
        state_action_values = logits.gather(1, a)

        # Compute V(s_{t+1}).
        with torch.no_grad():
            next_logits, h_t = self.target_net(next_o, next_z)
        next_state_values = (1 - d) * next_logits.max(1)[0].detach()
        # Compute the expected Q values.
        expected_state_action_values = (next_state_values * self.gamma) + r

        # Compute loss.
        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model.
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            if isinstance(param.grad, torch.Tensor):
                param.grad.data.clamp_(-1, 1)  # Constrain the gradient.
        self.optimizer.step()

        with torch.no_grad():
            for p, p_targ in zip(self.policy_net.parameters(), self.target_net.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
        info = {'q_loss': np.array(loss.item())}
        return info


def train_drqn(env, agent, batch_size=32, eps_start=1., eps_end=0.1, eps_decay=20000, update_every=1,
               num_train_episodes=500, test_freq=50, num_test_episodes=5, save_dir='./'):

    test_env = deepcopy(env)  # A copy of env
    logger = MetricLogger(save_dir)  # Metric logger holding training/test statistics
    logger.log_env_info(**env.get_env_info())

    max_epi_len = env.n_steps  # Maximum number of time steps in each episode
    steps_done = 0  # Time steps tha have elapsed
    tests_done = 0  # Number of tests have finished

    def test_agent():
        """Tests the performance of agents."""
        logger.init_test()
        for i_test in range(num_test_episodes):
            logger.init_episode()
            test_env.reset()
            obs, _, _, info = test_env.step(test_env.idle_act)  # Initial observation
            z = torch.zeros(agent.n_agents, agent.rnn_dim, device=device)  # Initial RNN hidden state
            logger.log_step(**info)
            for t in range(max_epi_len):
                # Agent selects an action following learnt policy.
                act, next_z = agent.select_action(obs, z, 0)
                # Receive reward and env transition.
                next_obs, rwd, done, info = test_env.step(act)
                # print(f"Ouside, env.p_uavs = {env.p_uavs}")
                logger.log_step(**info)
                logger.log_step(p_uavs=test_env.p_uavs.copy(), p_gts=test_env.p_gts.copy())
                # Move to the next observation.
                obs, z = next_obs, next_z
                # The current episode terminates when done=True.
                if done.any():
                    break
            logger.finish_test_episode()  # Current episode is finished.

    # Main loop of training
    start_time = time.time()  # Record the time of execution.

    for i_train in range(num_train_episodes):
        env.reset()  # Reset the positions of UAVs and GTs.
        logger.init_episode()
        obs = env.get_obs()  # Initial observation
        z = torch.zeros(agent.n_agents, agent.rnn_dim, device=device)  # Initial RNN hidden state
        for t in range(max_epi_len):
            # Select and perform an action.
            eps_thres = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
            # eps_thres = eps_end + (eps_start - eps_end) * steps_done / eps_decay
            act, next_z = agent.select_action(obs, z, eps_thres)  # Larger epsilon indicates better exploration
            # Receive reward and env transition.
            next_obs, rwd, done, info = env.step(act)
            logger.log_step(**info)
            # Store the transition in memory and update epi info.
            agent.cache(obs, z, act, next_obs, next_z, rwd, done)
            steps_done += 1
            # Move to the next observation.
            obs, z = next_obs, next_z
            # Perform one step of the optimization (on the policy network)
            if (steps_done % update_every == 0) and (len(agent.memory) >= batch_size):
                for _ in range(update_every):
                    train_info = agent.optimize(batch_size)  # Optimization starts when enough samples are collected.
                    logger.log_step(**train_info)
            # The current episode terminates when done is True.
            if done.any():
                break
        logger.finish_train_episode()  # Current episode is finished.

        # Test the performance of agents.
        if ((i_train + 1) % test_freq == 0) or ((i_train + 1) == num_train_episodes):
            test_agent()
            print(f"Complete {i_train + 1}/{num_train_episodes} training episodes at {datetime.datetime.now().strftime('D%m-%d_T%H-%M')}. "
                  f"{time.time() - start_time:.2f} sec passed since last update.")
            tests_done += 1  # Another test is finished.
            start_time = time.time()  # Reset the timer.

        # Call lr scheduler.
        if (i_train + 1) % max(1, int(num_train_episodes / 100)) == 0:
            agent.scheduler.step()
            # print(f"lr = {agent.optimizer.param_groups[0]['lr']}.")

    # Save all results to disk.
    logger.display_metrics(['r_glo'], 'train', 'sum', inr=1, win_size=10)
    logger.display_metrics(['r_glo'], 'test', 'sum', inr=test_freq, win_size=5)
    logger.save()
    print(f"Finish all works...")
