import sys
sys.path.append("./src")

import numpy as np
import matplotlib.pyplot as plt
from data_utils import gen_circle

mu = 0.012150584269940354


v1 = np.array([0, 1, 1])
v2 = np.array([1, 0, 0])

R = gen_circle(v1, v2, T = 2*np.pi,step=0.00645, r = 0.00477887617, center = [1-mu, 0, 0])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(R[:,0], R[:,1], R[:,2], marker='.', color="black")

ax.set_xlabel('X (DU)')
ax.set_ylabel('Y (DU)')
ax.set_zlabel('Z (DU)')

v2 = np.array([1, 1, 0])
R = gen_circle(v1, v2, T = 2*np.pi, step=0.00645, r = 0.00477887617, center = [1-mu, 0, 0])
ax.scatter(R[:,0], R[:,1], R[:,2], marker='.', color="red")

fname = "/Users/malavpatel/Research/SensorTasking/tmp/orbits/orbits.txt"
states = np.loadtxt(fname, delimiter=",")

pos = states[:,:3]
vel = states[:,3:7]
tspan = states[:,-1]

# ax.scatter(pos[:,0], pos[:,1], pos[:,2], marker='.', color="blue")

plt.show()