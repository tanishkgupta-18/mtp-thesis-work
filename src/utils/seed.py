import os
import random
import numpy as np
import tensorflow as tf


def set_seed(seed: int = 42) -> None:
    """Set random seed across Python, NumPy, and TensorFlow for reproducibility.

    Args:
        seed: The integer seed value to use.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
