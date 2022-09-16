import random
from copy import deepcopy

import torch as th
import torch.nn as nn
from torch.optim import AdamW

import dgl

from algos.common import *
from algos.drqn.buffer import ReplayBuffer
from algos.drqn.agents import REGISTRY as agent_REGISTRY


class QLearner:
    def __init__(self, env_info, args):
        self.args = args
        self.device = th.device(args.device)

        # Extract env info
        self.obs_shape = env_info['obs_shape']
        self.n_actions = env_info['n_actions']
        self.max_seq_len = args.max_seq_len if args.max_seq_len is not None else env_info['episode_limit']

        self.policy_net = self._build_agent().to(self.device)
        self.target_net = self._build_agent().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        print(f"policy network: \n{self.policy_net}")

        self.gamma = args.gamma
        self.polyak = args.polyak
        self.batch_size = args.batch_size

        self.buffer = ReplayBuffer(capacity=args.replay_size, max_seq_len=self.max_seq_len)

        self.loss_fn = nn.MSELoss()  # Loss function
        self.optimizer = AdamW(self.policy_net.parameters(), lr=args.lr)  # Optimizer

        self.anneal_lr = args.anneal_lr
        if self.anneal_lr:
            lr_lambda = lambda epoch: max(0.4, 1 - epoch / 100)
            self.lr_scheduler = th.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda, verbose=True)

    def init_hidden(self, batch_size=1):
        return self.policy_net.init_hidden().expand(batch_size, -1)

    def _build_agent(self):
        if isinstance(self.obs_shape, int):
            return agent_REGISTRY['rnn'](self.obs_shape, self.n_actions, self.args)
        else:
            return agent_REGISTRY['gnn'](self.obs_shape, self.n_actions, self.args)

    def act(self, obs, h, eps_thres):
        obs, h = obs.to(self.device), h.to(self.device)
        with th.no_grad():
            logits, h = self.policy_net(obs, h)

        # Select actions following epsilon-greedy strategy.
        if random.random() > eps_thres:
            act = th.argmax(logits, 1)  # Greedy actions
        else:
            act = th.randint(self.n_actions, size=(1,), dtype=th.long)  # Random action

        return act.item(), h

    def cache(self, obs, h, act, rew, next_obs, next_h, done, bad_mask):
        transition = dict(obs=obs, h=h,
                          act=th.tensor(act, dtype=th.long).reshape(1, 1),
                          rew=th.tensor(rew, dtype=th.float32).reshape(1, 1),
                          next_obs=next_obs, next_h=(1 - done) * next_h,
                          done=th.tensor((1 - bad_mask) * done, dtype=th.float32).reshape(1, 1))
        self.buffer.push(transition)

    def update(self):
        assert len(self.buffer) >= self.batch_size, "Insufficient samples for update."

        samples = self.buffer.sample(self.batch_size)  # List of sequences
        batch = {k: [] for k in self.buffer.scheme}

        for t in range(self.max_seq_len):
            for k in self.buffer.scheme:
                batch[k].append(cat([samples[i][k][t] for i in range(self.batch_size)]))

        for k in {'obs', 'h'}:
            if k in self.buffer.scheme:
                batch[k].append(cat([samples[i][k][self.max_seq_len] for i in range(self.batch_size)]))

        acts = th.stack(batch['act']).to(self.device)
        rews = th.stack(batch['rew']).to(self.device)
        dones = th.stack(batch['done']).to(self.device)
        h, h_targ = batch['h'][0].to(self.device), batch['h'][1].to(self.device)

        agent_out, target_out = [], []
        obs = [batch['obs'][t].to(self.device) for t in range(len(batch['obs']))]

        for t in range(self.max_seq_len):

            logits, h = self.policy_net(obs[t], h)
            state_action_values = logits.gather(1, acts[t])

            agent_out.append(state_action_values)
            with th.no_grad():
                next_logits, h_targ = self.target_net(obs[t + 1], h_targ)

            next_state_values = next_logits.max(1, keepdim=True)[0]
            target_out.append(next_state_values)

        q_est = th.stack(agent_out).view(self.max_seq_len, self.batch_size, 1)
        next_v = th.stack(target_out).view(self.max_seq_len, self.batch_size, 1)
        q_targ = rews + self.gamma * (1 - dones) * next_v
        loss = self.loss_fn(q_est, q_targ)

        # Perform one step of optimization.
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.policy_net.parameters(), clip_value=1)  # Gradient-clipping
        self.optimizer.step()

        # Update the target network via polyak averaging.
        with th.no_grad():
            for p, p_targ in zip(self.policy_net.parameters(), self.target_net.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        return dict(LossQ=loss.item(), QVals=q_est.detach().cpu().numpy())

    def save_checkpoint(self, path, stamp):
        """Saves checkpoint for inference or resuming training."""

        checkpoint = dict()
        checkpoint.update(stamp)
        checkpoint['model_state_dict'] = self.policy_net.state_dict()
        checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
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
        if self.anneal_lr:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        print(f"Load checkpoint from {path}.")
        return stamp
