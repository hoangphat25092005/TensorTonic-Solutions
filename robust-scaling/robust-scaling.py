import numpy as np

def robust_scaling(values):
    values = np.array(values)
    if len(values) == 1:
        return np.array([0.0])
    sorted_vals = np.sort(values)
    
    median = np.median(sorted_vals)
    
    n = len(sorted_vals)
    
    # Split data (exclude median if odd length)
    if n % 2 == 0:
        lower = sorted_vals[:n//2]
        upper = sorted_vals[n//2:]
    else:
        lower = sorted_vals[:n//2]
        upper = sorted_vals[n//2+1:]
    
    q1 = np.median(lower)
    q3 = np.median(upper)
    
    iqr = q3 - q1
    
    if iqr == 0:
        return values - median
    
    return (values - median) / iqr
