import numpy as np


def L2(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.subtract(pred, gt), ord=2, axis=-1)


def L1(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.subtract(pred, gt), ord=1, axis=-1)