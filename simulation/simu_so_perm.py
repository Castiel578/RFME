from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.learning.incremental_frechet_mean import IncrementalFrechetMean
from geomstats.learning.frechet_mean import FrechetMean

import time
import numpy as np
import backend_fun as fun

from scipy.spatial.transform import Rotation

t = 1000
n = 128
d = 3

so = SpecialOrthogonal(n = d)
mean_so = Rotation.from_matrix(np.eye(d))
        
mean_IFME = IncrementalFrechetMean(so)
mean_vec = FrechetMean(so)
mean_RFME_vec = fun.RecFrechetMean_Vec(so)

dist_IFME = np.zeros(t)
dist_RFME = np.zeros(t)
dist_FM = np.zeros(t)

rotvecs = 0.5 * np.random.randn(n, 3)
noise = Rotation.from_rotvec(rotvecs)
data = mean_so * noise
data = data.as_matrix()

for i in range(t):
    np.random.shuffle(data)

    mean_IFME.fit(data)
    mean_vec.fit(data)     
    mean_RFME_vec.fit(data)

    dist_IFME[i] = so.metric.dist(mean_so.as_matrix(), mean_IFME.estimate_)
    dist_RFME[i] = so.metric.dist(mean_so.as_matrix(), mean_RFME_vec.estimate_)
    dist_FM[i] = so.metric.dist(mean_so.as_matrix(), mean_vec.estimate_)
    

np.savez('Recursive/result/so_dist_perm.npz', FM = dist_FM, IFME = dist_IFME, RFME = dist_RFME)
