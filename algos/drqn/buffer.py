import random
from collections import deque


class ReplayBuffer:
    """Replay buffer holding episodes of fixed length"""

    scheme = ('obs', 'h', 'act', 'rew', 'done')

    def __init__(self, capacity, max_seq_len):
        self.memory = deque(maxlen=capacity)  # Memory holding data
        self.max_seq_len = max_seq_len  # Maximum length of episode

        self.curr_seq = {k: [] for k in self.scheme}  # Current episode
        self.ptr = 0  # Pointer of timesteps

    def push(self, transition: dict):
        for k, v in transition.items():
            if k in self.scheme:
                self.curr_seq[k].append(v)

        self.ptr += 1
        if self.ptr == self.max_seq_len:
            for k in {'obs', 'h'}:
                self.curr_seq[k].append(transition.get('next_' + k))

            self.memory.append(self.curr_seq)
            self.curr_seq = {k: [] for k in self.scheme}
            self.ptr = 0

    def sample(self, batch_size: int):
        """Selects a random batch of samples."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)