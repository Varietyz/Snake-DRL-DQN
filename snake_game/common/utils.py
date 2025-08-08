# common/utils.py
import numpy as np
def clean_input(X: np.ndarray) -> np.ndarray:
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
    if len(X.shape) == 1:
        X = X.reshape(1, -1)
    return X.astype(np.float32)
