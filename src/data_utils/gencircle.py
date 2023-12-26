import numpy as np

def gen_circle(v1, v2, T, r = 1, step = 0.01, center = np.array([0,0,0])):
    """
    Args
    v1 : first vector defining plane
    v2 : second vector defining plane
    T : period of the circle
    """

    w = 2*np.pi/T

    v1hat = v1 / np.linalg.norm(v1)
    v2hat = v2 / np.linalg.norm(v2)


    t = np.arange(0, T, step=step)

    R = np.zeros(shape=(len(t), 3))

    for i in range(3):
        R[:,i] = center[i] + r*np.cos(w * t)*v1hat[i] + r*np.sin(w * t)*v2hat[i]

    
    return R