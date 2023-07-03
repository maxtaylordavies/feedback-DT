from datetime import datetime
from itertools import accumulate
import os
import socket

import numpy as np
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper, RGBImgPartialObsWrapper
import torch
from tqdm import tqdm


def log(msg, outPath=None, with_tqdm=False):
    """Prints a message to the console and optionally writes it to a file.

    Args:
        msg (str): The message to print.
        outPath (str) (optional): The path to the file to write the message to.
    """
    msg = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {msg}"

    if with_tqdm:
        tqdm.write(msg)
    else:
        print(msg)

    if outPath:
        with open(outPath, "a+") as f:
            f.write(msg + "\n")


def setup_devices(seed, useGpu=True):
    useCuda = useGpu and torch.cuda.is_available()
    if useGpu and not useCuda:
        raise ValueError(
            "You wanted to use cuda but it is not available. "
            "Check nvidia-smi and your configuration. If you do "
            "not want to use cuda, pass the --no_gpu flag."
        )

    device = torch.device("cuda" if useCuda else "cpu")
    log(f"Using device: {torch.cuda.get_device_name()}")

    torch.manual_seed(seed)

    if useCuda:
        device_str = f"{device.type}:{device.index}" if device.index else f"{device.type}"
        os.environ["CUDA_VISIBLE_DEVICES"] = device_str
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        # This does make things slower :(
        torch.backends.cudnn.benchmark = False

    return device


def is_network_connection():
    host, port, timeout = "8.8.8.8", 53, 3
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error as ex:
        log(f"Network connection error: {ex}")
        return False


def to_one_hot(x, width=None):
    if width:
        res = np.zeros((x.size, width))
        res[np.arange(x.size), x] = 1
    else:
        res = torch.zeros_like(x)
        res[x.argmax()] = 1
    return res


def discounted_cumsum(x, gamma=1):
    return np.array(list(accumulate(x[::-1], lambda a, b: (gamma * a) + b)))[::-1]


def name_dataset(args):
    return f"{args['env_name']}_{args['num_episodes']}-eps_{'incl' if args['include_timeout'] else 'excl'}-timeout"


def get_minigrid_obs(env, partial_obs, fully_obs=False, rgb_obs=False):
    """
    Get the observation from the environment.

    Parameters
    ----------
    partial_observation (np.ndarray): the partial observation from the environment.
    env (gym.Env): the environment.

    Returns
    -------
    np.ndarray: the observation, either as a symbolic or rgb image representation.
    """
    if fully_obs and rgb_obs:
        _env = RGBImgObsWrapper(env)
        return _env.observation({})
    elif fully_obs and not rgb_obs:
        _env = FullyObsWrapper(env)
        return _env.observation({})
    elif not fully_obs and rgb_obs:
        _env = RGBImgPartialObsWrapper(env)
        return _env.observation({})
    else:
        return partial_obs
