import numpy as np

def dice_loss(p, y, eps=1e-8):
  p = np.array(p, dtype=np.float32)
  y = np.array(y, dtype=np.float32)
  #Just accept 1D and 2D input
  #1D input case
  if len(p.shape) == 1:
    dice = ((2 * np.sum(p * y)) + eps) / ((np.sum(p)) + (np.sum(y)) + eps)
    return 1 - dice
  #2D input case
  elif len(p.shape) == 2:
    dice = np.zeros(p.shape[0])
    for i in range(p.shape[0]):
      dice[i] = ((2 * np.sum(p[i] * y[i])) + eps) / ((np.sum(p[i])) + (np.sum(y[i])) + eps)
    return 1 - np.mean(dice)