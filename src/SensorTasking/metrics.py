import numpy as np


def distance_helper(rO, rT, cutoff):

    dist = np.linalg.norm(rO - rT)

    flag = False
    if dist <= cutoff:
        flag = True
    else:
        flag = False
    
    return flag

def occlude_helper(rO, rT, cutoff, rB):


    rOB = rB - rO

    w1 = [0, -rOB[2], rOB[1]] / np.linalg.norm([0, -rOB[2], rOB[1]])

    b1 = np.dot(w1, rB)

    # w2 normalized to unit length
    w2 = np.cross(rOB, w1)/ np.linalg.norm(rOB)

    # b2 offset
    b2 = np.dot(w2, rB)

    # w3 normalized to unit length
    w3 = rOB/np.linalg.norm(rOB)

    # b3 offset
    b3 = np.dot(w3, rB)

    return not((np.abs(np.dot(w1, rT) - b1) <= cutoff) and (np.abs(np.dot(w2, rT) - b2) <= cutoff) and (np.dot(w3, rT) - b3 > 0))
    




class Metric:

    def __init__(self, metric, cutoff, params = {}):
        
        self.metric = metric
        self.cutoff = cutoff
        self.params = params

    def apply(self, rO, rT):

        value = self.metric(rO, rT, self.cutoff,  **self.params)

        return value
    
    def applyv(self, rO, rTs):
        num_rTs = rTs.shape[0]

        values = np.zeros(shape=num_rTs, dtype=bool)

        for i in range(num_rTs):
            values[i] = self.apply(rO, rTs[i, :])

        return values



