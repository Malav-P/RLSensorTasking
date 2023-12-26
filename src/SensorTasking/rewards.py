import numpy as np

def asymmetric_gaussian(x, mean = 20, peak_reward = 5, baseline_left = 2, baseline_right = 1, sigma_left = 9.4, sigma_right = 1):
    
    if x <= mean:
        ans = np.exp(-((x-mean)**2) / (2*sigma_left**2)) * (peak_reward - baseline_left) + baseline_left
    else:
        ans =  np.exp(-((x-mean)**2) / (2*sigma_right**2)) * (peak_reward - baseline_right) + baseline_right


    return ans