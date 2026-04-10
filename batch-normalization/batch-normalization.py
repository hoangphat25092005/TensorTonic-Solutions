import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
  """ Convert input to numpy array including beta, gamma, and inputs x"""
  x = np.array(x)
  beta = np.array(beta)
  gamma = np.array(gamma)

  """If inputs shaped (N, D): Normalize each feature over the batch axis (axis=0)
     (N, C, H, W): Normalize each channel over axes (0, 2, 3)"""
  if len(x.shape) == 2:
    #Calculation steps
    cal_mu = np.mean(x, axis=0)
    cal_var = np.mean((x - cal_mu) ** 2, axis=0) 
    x = (x - cal_mu) / (np.sqrt(cal_var + eps))
    out = gamma * x + beta
  else:
    # Reshape gamma and beta for broadcasting with 4D input (N, C, H, W)
    gamma = gamma.reshape(1, -1, 1, 1)
    beta = beta.reshape(1, -1, 1, 1)
    cal_mu = np.mean(x, axis=(0, 2, 3), keepdims=True)
    cal_var = np.mean((x - cal_mu) ** 2, axis=(0, 2, 3), keepdims=True)
    x = (x - cal_mu) / (np.sqrt(cal_var + eps))
    out = gamma * x + beta
  return out