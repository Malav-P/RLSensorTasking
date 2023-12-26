import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

def load_horizons(fname):
    
    start_line = "$$SOE"
    end_line = "$$EOE"

    count = 0
    span = []
    with open(fname, 'r') as file:
        # Read each line in the file
        for line in file:
            count += 1
            if start_line in line or end_line in line:
                span.append(count)

    data = np.loadtxt(fname, skiprows=span[0], max_rows=span[1] - span[0] - 1, delimiter=',', usecols=[0, 2, 3, 4])

    return data

def plot_components(data):
    
    fig, ax = plt.subplots(3, 1)
    
    ax[0].scatter(data[:,0], data[:,1])
    ax[0].set_ylabel("x")

    ax[1].scatter(data[:,0], data[:,2])
    ax[1].set_ylabel("y")

    ax[2].scatter(data[:,0], data[:,3])
    ax[2].set_ylabel("z")

    ax[2].set_xlabel("time (TU)")

    plt.tight_layout()

    return fig, ax

def find_periodicity(data, xtol = 0.001, ytol = 0.001, ztol = 0.001):

    subtracted = data - data[0, :]

    x_rep = np.where(np.abs(subtracted[:,1]/np.max(np.abs(subtracted[:,1]))) < xtol)
    y_rep = np.where(np.abs(subtracted[:,2]/np.max(np.abs(subtracted[:,2]))) < ytol)
    z_rep = np.where(np.abs(subtracted[:,3]/np.max(np.abs(subtracted[:,3]))) < ztol)

    return x_rep, y_rep, z_rep

def normalize_data(data, LU = 1.0, TU = 1.0, center = np.array([0, 0, 0])):
    norm_data = np.copy(data)
    norm_data[:, 0] = (data[:,0] - data[0, 0]) / TU
    norm_data[:, [1, 2, 3]] = data[:, [1, 2, 3]] / LU + np.tile(center, (data.shape[0], 1))
    

    return norm_data

def make_spline(norm_data):
    x = np.append(norm_data[:,0], norm_data[-1, 0] + norm_data[1,0])
    y = np.append(norm_data[:, [1,2,3]], [norm_data[0, [1, 2, 3]]], axis=0)

    bspl = make_interp_spline(x, y, k=3, bc_type='periodic', axis=0)

    return bspl

def gen_target(bspl, step, stop, start = 0):
    tt = np.arange(start=start, step=step, stop=stop)

    R = bspl(tt)

    return R

