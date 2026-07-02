from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.incremental_frechet_mean import IncrementalFrechetMean
from geomstats.learning.frechet_mean import FrechetMean
 
import numpy as np
import matplotlib.pyplot as plt
import time

import backend_fun as fun

def sampling_sphere(n_sample):
    #phi = np.random.uniform(1.2*np.pi, 2.2*np.pi, n_sample)
    phi = np.random.uniform(0*np.pi, 1*np.pi, n_sample)
    costheta = np.random.uniform(-1, 1, n_sample)
    theta = np.arccos(costheta)

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    samples = np.column_stack((x,y,z))
    
    return samples
    
#np.random.seed(27)
#np.random.seed(43)
t = 1000
n = np.arange(500, 5001, 500)
d = 2
mu = np.array((0, 1, 0))

sphere = Hypersphere(dim = 2)

mean_vec = FrechetMean(sphere)
mean_RFME_vec = fun.RecFrechetMean_Vec(sphere)

time_FM_vec = np.zeros((len(n), t))
time_RFME_vec = np.zeros((len(n), t))

for i in range(len(n)):
    for j in range(t):
        data = sampling_sphere(n[i])

        start_time = time.perf_counter()
        mean_vec.fit(data)
        time_FM_vec[i][j] = (time.perf_counter() - start_time)

        start_time = time.perf_counter()
        mean_RFME_vec.fit(data)
        time_RFME_vec[i][j] = (time.perf_counter() - start_time)

np.savez('Recursive/result/sphere_time_vec.npz', FMvec = time_FM_vec, RFMEvec = time_RFME_vec)
