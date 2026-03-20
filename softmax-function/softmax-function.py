import numpy as np

def softmax(x):
    x = np.array(x)

    if x.ndim == 1:
        # 1D case
        x_max = np.max(x)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x)

    elif x.ndim == 2:
        # 2D case (row-wise)
        x_max = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    else:
        raise ValueError("Input must be 1D or 2D")