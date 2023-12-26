import sys
sys.path.append("./src")

import numpy as np
import matplotlib.pyplot as plt
from SensorTasking import asymmetric_gaussian

t = np.linspace(-1, 30, 300)
y = [asymmetric_gaussian(i) for i in t]

plt.scatter(t, y, marker='.')
plt.show()
