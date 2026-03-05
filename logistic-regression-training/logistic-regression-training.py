import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.array(X)
    y = np.array(y)
    
    w = np.zeros(X.shape[1]) 
    b = 0

    for i in range(steps):
        z = np.dot(X, w) + b
        y_pred = _sigmoid(z)
        
        # add an epsilon to prevent log(0)
        eps = 1e-15
        L = -(1/X.shape[0]) * np.sum(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))

        # Calculate gradient
        dw = (1 / X.shape[0]) * np.dot(X.T, (y_pred - y))
        db = (1 / X.shape[0]) * np.sum(y_pred - y)

        w -= lr * dw
        b -= lr * db

    return (w, b)