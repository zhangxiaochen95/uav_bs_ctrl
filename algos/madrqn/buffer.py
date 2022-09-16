import random
from collections import deque

scheme = ('obs', 'h', 'state', 'act', 'rew', 'done')


class ReplayBuffer(object):
    """Replay buffer for recurrent agents"""

    def __init__(self, capacity, max_seq_len, scheme=scheme):
        self.memory = deque(maxlen=capacity)  # Memory holding data
        self.max_seq_len = max_seq_len  # Maximum length of time sequence
        self.scheme = scheme  # Keys of entries

        self.curr_seq = {k: [] for k in self.scheme}  # Current sequence
        self.ptr = 0  # Pointer of sequence.

    def push(self, transition: dict):
        """Stores a transition to memory."""
        # Put each entry of transition into sequence.
        for k, v in transition.items():
            if k in self.scheme:
                self.curr_seq[k].append(v)
        self.ptr += 1

        # When sufficient transitions are collected for sequence.
        if self.ptr == self.max_seq_len:
            # Append the next obs/h/state of the last timestep.
            for k in {'obs', 'h', 'state'}:
                if k in self.scheme:
                    self.curr_seq[k].append(transition.get('next_' + k))

            self.memory.append(self.curr_seq)  # Put sequence to memory.
            self.curr_seq = {k: [] for k in self.scheme}  # Dump sequence.
            self.ptr = 0  # Reset pointer.

    def sample(self, batch_size: int):
        """Selects a random batch of samples."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
