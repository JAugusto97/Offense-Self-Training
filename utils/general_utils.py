import random
from typing import Optional
import numpy as np
import torch
import logging


def get_logger(level: Optional[str] = "debug", filename: Optional[str] = None) -> logging.Logger:
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "fatal": logging.FATAL,
    }
    loglevel = level_map.get(level)
    if loglevel is None:
        raise TypeError

    logging.basicConfig(
        level=loglevel,
        filename=filename,
        filemode="w" if filename else None,
        format="%(levelname)s | %(asctime)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )

    logger = logging.getLogger()
    return logger


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
