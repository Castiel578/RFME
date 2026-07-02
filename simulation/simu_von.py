from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.incremental_frechet_mean import IncrementalFrechetMean
from geomstats.learning.frechet_mean import FrechetMean

import time
import numpy as np
import backend_fun as fun

#np.random.seed(20)
t = 1000
n = 1000
d = 2
kappas = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])

sphere = Hypersphere(dim = 2)
mu = np.array((0,1,0))

mean_RFME = fun.RecFrechetMean(sphere)
mean_IFME = IncrementalFrechetMean(sphere)
mean_m = fun.FrechetMean_m(sphere)
mean_vec = FrechetMean(sphere)
mean_RFME_vec = fun.RecFrechetMean_Vec(sphere)

dist_IFME = np.zeros((len(kappas), t))
dist_RFME = np.zeros((len(kappas), t))
dist_FM = np.zeros((len(kappas), t))
time_FM = np.zeros(((len(kappas), t)))
time_IFME = np.zeros((len(kappas), t))
time_RFME = np.zeros((len(kappas), t))
time_FM_vec = np.zeros((len(kappas), t))
time_RFME_vec = np.zeros((len(kappas), t))


for i in range(len(kappas)):
    for j in range(t):
        data = sphere.random_von_mises_fisher(mu ,kappa = 1 / kappas[i], n_samples = n)

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

        dist_FM[i][j] = sphere.metric.dist(mu, mean_m.estimate_)
        dist_IFME[i][j] = sphere.metric.dist(mu, mean_IFME.estimate_)
        dist_RFME[i][j] = sphere.metric.dist(mu, mean_RFME.estimate_)

        
np.savez('Recursive/result/von_error.npz', FM = dist_FM, IFME = dist_IFME, RFME = dist_RFME)
np.savez('Recursive/result/von_time.npz', FM = time_FM, IFME = time_IFME, RFME = time_RFME, FMvec = time_FM_vec, RFMEvec = time_RFME_vec)