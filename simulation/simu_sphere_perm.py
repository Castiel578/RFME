from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.incremental_frechet_mean import IncrementalFrechetMean
from geomstats.learning.frechet_mean import FrechetMean
 
import numpy as np

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
    
t = 1000
n = 512
d = 2
mu = np.array((0, 1, 0))

sphere = Hypersphere(dim = 2)

mean_IFME = IncrementalFrechetMean(sphere)
mean_vec = FrechetMean(sphere)
mean_RFME_vec = fun.RecFrechetMean_Vec(sphere)

dist_IFME = np.zeros(t)
dist_RFME = np.zeros(t)
dist_FM = np.zeros(t)

data = sampling_sphere(n)

for i in range(t):
    np.random.shuffle(data)

    mean_IFME.fit(data)
    mean_vec.fit(data)
    mean_RFME_vec.fit(data)

    dist_FM[i] = sphere.metric.dist(mu, mean_vec.estimate_)
    dist_IFME[i] = sphere.metric.dist(mu, mean_IFME.estimate_)
    dist_RFME[i] = sphere.metric.dist(mu, mean_RFME_vec.estimate_)


np.savez('Recursive/result/sphere_dist_perm.npz', FM = dist_FM, IFME = dist_IFME, RFME = dist_RFME)
