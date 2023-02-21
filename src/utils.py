from datetime import datetime
import os

import torch


def log(msg, outPath=None):
    """Prints a message to the console and optionally writes it to a file.

    Args:
        msg (str): The message to print.
        outPath (str) (optional): The path to the file to write the message to.
    """
    msg = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {msg}"
    print(msg)
    if outPath:
        with open(outPath, "a+") as f:
            f.write(msg + "\n")


def setup_devices(useGpu=True, seed=None):
    useCuda = useGpu and torch.cuda.is_available()
    if useGpu and not useCuda:
        raise ValueError(
            "You wanted to use cuda but it is not available. "
            "Check nvidia-smi and your configuration. If you do "
            "not want to use cuda, pass the --no_gpu flag."
        )

    device = torch.device("cuda" if useCuda else "cpu")
    log(f"Using device: {torch.cuda.get_device_name()}")

    if not seed:
        seed = torch.randint(0, 2**32, (1,)).item()
        log(f"You did not set seed, so {seed} was chosen")

    torch.manual_seed(seed)
    if useCuda:
        device_str = (
            f"{device.type}:{device.index}" if device.index else f"{device.type}"
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = device_str
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        # This does make things slower :(
        torch.backends.cudnn.benchmark = False
