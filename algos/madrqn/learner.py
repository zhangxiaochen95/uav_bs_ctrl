import random
from copy import deepcopy

import torch as th
import torch.nn as nn
from torch.optim import AdamW

from algos.common import *
from algos.madrqn.buffer import ReplayBuffer
from algos.madrqn.agents import REGISTRY as agent_REGISTRY
from algos.madrqn.agents.mixers import QMixer


class MultiAgentQLearner:
    """Multi-agent Q learner"""

    def __init__(self, env_info, args):
        self.args = args
        self.device = th.device(args.device)

        # Extract env info
        self.obs_shape = env_info['obs_shape']
        self.state_shape = env_info['state_shape']
        self.n_actions = env_info['n_actions']
        self.n_agents = env_info['n_agents']

        self.policy_net = self._build_agent().to(self.device)  # Policy network
        self.target_net = self._build_agent().to(self.device)  # Target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        print(f"policy network: \n{self.policy_net}")
        self.params = list(self.policy_net.parameters())  # Parameters to optimize

        # QMix
        self.mixer = None
        if args.mixer:
            self.mixer = QMixer(self.state_shape, self.n_agents, args).to(self.device)  # QMixer
            self.target_mixer = deepcopy(self.mixer).to(self.device)
            print(f"mixer = \n{self.mixer}")
            self.params += list(self.mixer.parameters())

        self.max_seq_len = args.max_seq_len if args.max_seq_len is not None else env_info['episode_limit']
        self.gamma = args.gamma  # Discount factor
        self.polyak = args.polyak  # Interpolation factor in polyak averaging for target networks
        self.batch_size = args.batch_size  # Mini-batch size for SGD

        self.buffer = ReplayBuffer(args.replay_size, self.max_seq_len)  # Replay buffer
        self.loss_fn = nn.MSELoss()  # Loss function
        self.optimizer = AdamW(self.params, lr=args.lr)  # Optimizer

        self.anneal_lr = args.anneal_lr  # Whether lr annealing is used.
        if self.anneal_lr:
            lr_lambda = lambda epoch: max(0.4, 1 - epoch / 100)
            self.lr_scheduler = th.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda, verbose=True)

        self.double_q = args.double_q

    def init_hidden(self, batch_size=1):
        """Initializes RNN hidden states for all agents."""
        return self.policy_net.init_hidden().expand(self.n_agents * batch_size, -1)

    def _build_agent(self):
        """Builds agents."""
        if (self.args.o == 'mlp') and (self.args.c is None):
            return agent_REGISTRY['rnn'](self.obs_shape, self.n_actions, self.args)
        else:
            return agent_REGISTRY['gnn'](self.obs_shape, self.n_actions, self.args)

    def act(self, obs, h, eps_thres):
        """Selects actions following epsilon-greedy strategy."""
        obs, h = obs.to(self.device), h.to(self.device)
        with th.no_grad():
            logits, h = self.policy_net(obs, h)

        if random.random() > eps_thres:
            acts = th.argmax(logits, 1)  # Greedy actions
        else:
            acts = th.randint(self.n_actions, size=(self.n_agents,), dtype=th.long)  # Random actions

        return acts.tolist(), h

    def cache(self, obs, h, state, act, rew, next_obs, next_h, next_state, done, bad_mask):
        if self.args.share_reward:
            rew = rew.mean()

        # When done is True due to reaching episode limit, mute it.
        transition = dict(obs=obs, h=h, state=state,
                          act=th.tensor(act, dtype=th.long).unsqueeze(1),
                          rew=th.tensor(rew, dtype=th.float32).reshape(1, -1),
                          next_obs=next_obs, next_h=(1 - done) * next_h, next_state=next_state,
                          done=th.tensor((1 - bad_mask) * done, dtype=th.float32).reshape(1, 1))
        self.buffer.push(transition)

    def update(self):
        """Updates parameters of recurrent agents via BPTT."""

        assert len(self.buffer) >= self.batch_size, "Insufficient samples for update."

        samples = self.buffer.sample(self.batch_size)  # List of sequences
        batch = {k: [] for k in self.buffer.scheme}  # Dict holding batch of samples.

        # Construct input sequences.
        for t in range(self.max_seq_len):
            for k in batch.keys():
                batch[k].append(cat([samples[i][k][t] for i in range(self.batch_size)]))
        # Append next obs/h/state of the last timestep.
        for k in {'obs', 'h', 'state'}:
            batch[k].append(cat([samples[i][k][self.max_seq_len] for i in range(self.batch_size)]))

        acts = th.stack(batch['act']).to(self.device)
        rews = th.stack(batch['rew']).to(self.device)
        dones = th.stack(batch['done']).to(self.device)
        h, h_targ = batch['h'][0].to(self.device), batch['h'][1].to(self.device)  # Get initial hidden states.

        agent_out, target_out = [], []
        obs = [batch['obs'][t].to(self.device) for t in range(len(batch['obs']))]

        for t in range(self.max_seq_len):
            # Policy network predicts the Q(s_{t},a_{t}) at current timestep.
            logits, h = self.policy_net(obs[t], h)
            agent_out.append(logits)
            # Target network predicts Q(s_{t+1}, a_{t+1}).
            with th.no_grad():
                next_logits, h_targ = self.target_net(obs[t + 1], h_targ)
                target_out.append(next_logits)

        # Let policy network make predictions for next state of the last timestep in the sequence.
        logits, h = self.policy_net(obs[self.max_seq_len], h)
        agent_out.append(logits)
        # Stack outputs of policy/target networks.
        agent_out, target_out = th.stack(agent_out), th.stack(target_out)

        # Compute Q_{s_{t}, a_{t}} with policy network.
        qvals = agent_out[:-1].gather(2, acts)
        # Compute V_{s_{t+1}}.
        if not self.double_q:
            next_vals = target_out.max(2, keepdim=True)[0]
        else:
            next_acts = th.argmax(agent_out[1:].clone().detach(), 2, keepdims=True)
            next_vals = target_out.gather(2, next_acts)

        qvals = qvals.view(self.max_seq_len, self.batch_size, self.n_agents)
        next_vals = next_vals.view(self.max_seq_len, self.batch_size, self.n_agents)
        # Compute centralized Q values when QMix is used.
        if self.mixer is not None:
            states = th.stack(batch['state']).to(self.device)
            qvals = self.mixer(qvals, states[:-1])
            next_vals = self.target_mixer(next_vals, states[1:])

        # Obtain target of update.
        rews, dones = rews.expand_as(next_vals), dones.expand_as(next_vals)
        target_qvals = rews + self.gamma * (1 - dones) * next_vals
        # Compute MSE loss.
        loss = self.loss_fn(qvals, target_qvals)

        # Call one step of gradient descent.
        self.optimizer.zero_grad()
        loss.backward()  # Back propagation
        nn.utils.clip_grad_value_(self.policy_net.parameters(), clip_value=1)  # Gradient-clipping
        self.optimizer.step()  # Call update.

        # Update the target network via polyak averaging.
        with th.no_grad():
            for p, p_targ in zip(self.policy_net.parameters(), self.target_net.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

            if self.mixer is not None:
                for p, p_targ in zip(self.mixer.parameters(), self.target_mixer.parameters()):
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

        return dict(LossQ=loss.item(), QVals=qvals.detach().cpu().numpy())

    def save_checkpoint(self, path, stamp):
        """Saves checkpoint for inference or resuming training."""
        checkpoint = dict()
        checkpoint.update(stamp)
        checkpoint['model_state_dict'] = self.policy_net.state_dict()
        checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        if self.mixer is not None:
            checkpoint['mixer_state_dict'] = self.mixer.state_dict()
        if self.anneal_lr:
            checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()
        th.save(checkpoint, path)
        print(f"Save checkpoint to {path}.")

    def load_checkpoint(self, path):
        """Loads checkpoint from given path."""
        checkpoint = th.load(path, map_location=self.device)
        stamp = dict(epoch=checkpoint['epoch'], t=checkpoint['t'])
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.mixer is not None:
            self.mixer.load_state_dict(checkpoint['mixer_state_dict'])
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        if self.anneal_lr:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        print(f"Load checkpoint from {path}.")
        return stamp
