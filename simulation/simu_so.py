from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.learning.incremental_frechet_mean import IncrementalFrechetMean
from geomstats.learning.frechet_mean import FrechetMean

import time
import numpy as np
import backend_fun as fun

import matplotlib.pyplot as plt
import geomstats.visualization as vs

from scipy.spatial.transform import Rotation

#np.random.seed(24)
#np.random.seed(37)
t = 500
n = np.arange(50, 501, 50)
d = 3

so = SpecialOrthogonal(n = d)
mean_so = Rotation.from_matrix(np.eye(3))
        
mean_RFME = fun.RecFrechetMean(so)
mean_m = fun.FrechetMean_m(so)
mean_IFME = IncrementalFrechetMean(so)
mean_vec = FrechetMean(so)
mean_RFME_vec = fun.RecFrechetMean_Vec(so)

dist_IFME = np.zeros((len(n), t))
dist_RFME = np.zeros((len(n), t))
dist_FM = np.zeros((len(n), t))
time_FM = np.zeros(((len(n), t)))
time_IFME = np.zeros((len(n), t))
time_RFME = np.zeros((len(n), t))
time_FM_vec = np.zeros((len(n), t))
time_RFME_vec = np.zeros((len(n), t))


for i in range(len(n)):
    for j in range(t):
        rotvecs = 0.5 * np.random.randn(n[i], 3)
        noise = Rotation.from_rotvec(rotvecs)
        data = mean_so * noise
        data = data.as_matrix()
        #data = so.random_point(n_samples = n[i])

        start_time = time.perf_counter()
        mean_IFME.fit(data)
        time_IFME[i][j] = (time.perf_counter() - start_time)

        start_time = time.perf_counter()
        mean_RFME.fit(data)
        time_RFME[i][j] = (time.perf_counter() - start_time)

        start_time = time.perf_counter()
        mean_m.fit(data)
        time_FM[i][j] = (time.perf_counter() - start_time)

        start_time = time.perf_counter()
        mean_vec.fit(data)
        time_FM_vec[i][j] = (time.perf_counter() - start_time)

        start_time = time.perf_counter()
        mean_RFME_vec.fit(data)
        time_RFME_vec[i][j] = (time.perf_counter() - start_time)

        dist_IFME[i][j] = so.metric.dist(mean_so.as_matrix(), mean_IFME.estimate_)
        dist_RFME[i][j] = so.metric.dist(mean_so.as_matrix(), mean_RFME.estimate_)
        dist_FM[i][j] = so.metric.dist(mean_so.as_matrix(), mean_m.estimate_)
    

np.savez('Recursive/result/so_error.npz', FM = dist_FM, IFME = dist_IFME, RFME = dist_RFME)
np.savez('Recursive/result/so_time.npz', FM = time_FM, IFME = time_IFME, RFME = time_RFME, FMvec = time_FM_vec, RFMEvec = time_RFME_vec)
