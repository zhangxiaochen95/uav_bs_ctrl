import random
import numpy as np
import torch as th
import torch as th
import dgl


# ==================================================================================================================== #
# Functions used by run functions.

def check_args_sanity(args):
    """Checks sanity and avoids conflicts of arguments."""

    # Ensure specified cuda is used when it is available.
    if args.device == 'cuda' and th.cuda.is_available():
        args.device = f'cuda:{args.cuda_index}'
    else:
        args.device = 'cpu'
    print(f"Choose to use {args.device}.")

    # When QMix is used, ensure a scalar reward is used.
    if hasattr(args, 'mixer'):
        if args.mixer and not args.share_reward:
            args.share_reward = True
            print("Since QMix is used, all agents are forced to share a scalar reward.")

    return args


def set_rand_seed(seed):
    """Sets random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

# ==================================================================================================================== #
# Functions used by learners


def cat(data_list):
    """Concatenates list of inputs"""
    if isinstance(data_list[0], th.Tensor):
        return th.cat(data_list)
    elif isinstance(data_list[0], dgl.DGLGraph):
        return dgl.batch(data_list)
    else:
        raise TypeError("Unrecognised observation type.")
