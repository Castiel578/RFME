import numpy as np
import math

from sklearn.base import BaseEstimator
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.learning.frechet_mean import BaseGradientDescent, GradientDescent
import geomstats.backend as gs
import logging
import geomstats.errors as error

class RecFrechetMean(BaseEstimator):

    def __init__(
            self,
            space,
    ):
        self.space = space
        self.estimate_ = None

    def fit(self, X, weights = None):
        
        if X.shape[0] == 1:
            self.estimate_ = X
            return self

        m_rec = X
        if weights is None:
            weights = np.full(X.shape[0], 1)
        
        N = m_rec.shape[0]
        k = N - 2 ** math.floor(math.log(N, 2))
        if k != 0:
            rec_new = np.empty((k, *m_rec.shape[1:]))
            weights_new = np.empty(k)
            for i in range(k):
                v = self.space.metric.log(m_rec[2*i+1], m_rec[2*i])
                rec_new[i] = self.space.metric.exp((weights[2*i+1] / (weights[2*i] + weights[2*i+1]))*v, m_rec[2*i])
                # geod_func = self.space.metric.geodesic(m_rec[2*i], m_rec[2*i+1])
                # rec_new[i] = geod_func(weights[2*i+1] / (weights[2*i] + weights[2*i+1]))[0]
                weights_new[i] = weights[2*i] + weights[2*i+1]
            m_rec = np.concatenate((rec_new, m_rec[2*k:]), axis = 0)
            weights = np.concatenate((weights_new, weights[2*k:]), axis = 0)                           

        while(True):
            N = m_rec.shape[0]
            weights_new = np.empty(N//2)
            rec_new = np.empty((N//2, *m_rec.shape[1:]))
            
            for i in range(N//2):
                v = self.space.metric.log(m_rec[2*i+1], m_rec[2*i])
                rec_new[i] = self.space.metric.exp((weights[2*i+1] / (weights[2*i] + weights[2*i+1]))*v, m_rec[2*i])
                # geod_func = self.space.metric.geodesic(m_rec[2*i], m_rec[2*i+1])
                # rec_new[i] = geod_func(weights[2*i+1] / (weights[2*i] + weights[2*i+1]))[0]
                weights_new[i] = weights[2*i] + weights[2*i+1]
            
            if N == 2:
                self.estimate_ = rec_new[0]
                break
            
            m_rec = rec_new
            weights = weights_new
        
        return self

def _scalarmulsum(scalar, array):
    return gs.einsum("n,n...->...", scalar, array)

class GradientDescent_m(BaseGradientDescent):

    def minimize(self, space, points, weights=None):
        n_points = gs.shape(points)[0]
        if weights is None:
            weights = gs.ones((n_points,))

        mean = points[0] if self.init_point is None else self.init_point

        if n_points == 1:
            return mean
        
        sum_weights = gs.sum(weights)
        iteration = 0
        sq_dist = 0.0
        var = 0.0

        norm_old = gs.linalg.norm(points)
        step_size = self.init_step_size
        logs = gs.zeros(gs.shape(points))

        while iteration < self.max_iter:
            for i in range(n_points):
                logs[i] = space.metric.log(point = points[i], base_point=mean)

            #logs = space.metric.log(point=points, base_point=mean)

            var = gs.sum(space.metric.squared_norm(logs, mean) * weights) / sum_weights

            tangent_mean = _scalarmulsum(weights, logs)
            tangent_mean /= sum_weights
            norm = gs.linalg.norm(tangent_mean)

            sq_dist = space.metric.squared_norm(tangent_mean, mean)

            var_is_0 = gs.isclose(var, 0.0)

            sq_dist_is_small = gs.less_equal(sq_dist, self.epsilon * space.dim)

            condition = ~gs.logical_or(var_is_0, sq_dist_is_small)
            if not (condition or iteration == 0):
                break

            estimate_next = space.metric.exp(step_size * tangent_mean, mean)
            mean = estimate_next
            iteration += 1

            if norm < norm_old:
                norm_old = norm
            elif norm > norm_old:
                step_size = step_size / 2.0

        if iteration == self.max_iter:
            logging.warning(
                "Maximum number of iterations %d reached. The mean may be inaccurate",
                self.max_iter,
            )

        if self.verbose:
            logging.info(
                "n_iter: {}, final variance: {}, final dist: {}".format(
                    iteration, var, sq_dist
                )
            )
        
        return mean
    
class FrechetMean_m(FrechetMean):

    def __new__(cls, space, **kwargs):
        return super().__new__(cls, space, **kwargs)

    def __init__(self, space, method="default"):
        super().__init__(space, method)
    
    def set(self, **kwargs):
        return super().set(**kwargs)

    @property
    def method(self):
        return self._method
    
    @method.setter
    def method(self, value):
        """Gradient descent method."""
        error.check_parameter_accepted_values(
            value, "method", ["default"]
        )
        if value == self._method:
            return

        self._method = value
        MAP_OPTIMIZER = {
            "default": GradientDescent_m,

        }
        self.optimizer = MAP_OPTIMIZER[value]()

    def fit(self, X, y=None, weights=None):
        self.estimate_ = self.optimizer.minimize(
            space = self.space,
            points=X,
            weights=weights
        )
        return self

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
    
class RecFrechetMean_Vec(BaseEstimator):

    def __init__(
            self,
            space,
    ):
        self.space = space
        self.estimate_ = None

    def fit(self, X, weights = None):
        
        if X.shape[0] == 1:
            self.estimate_ = X
            return self

        m_rec = X
        if weights is None:
            weights = np.full(X.shape[0], 1)
        
        N = m_rec.shape[0]
        k = N - 2 ** math.floor(math.log(N, 2))
        if k != 0:
            rec_p = m_rec[:2*k:2]
            rec_q = m_rec[1:2*k:2]
            weights_p = weights[:2*k:2]
            weights_q = weights[1:2*k:2]
            weights_new = weights_q + weights_p
            weights_ratio = weights_q / weights_new
            
            vs = self.space.metric.log(rec_q, rec_p)
            expand_dims = (1,) * (vs.ndim - 1)
            weights_ratio = weights_ratio.reshape((k, *expand_dims))
            rec_new = self.space.metric.exp(weights_ratio * vs, rec_p)
            m_rec = np.concatenate((rec_new, m_rec[2*k:]), axis = 0)
            weights = np.concatenate((weights_new, weights[2*k:]), axis = 0)                           

        while(True):
            N = m_rec.shape[0]
            rec_p = m_rec[:N:2]
            rec_q = m_rec[1:N:2]
            weights_p = weights[:N:2]
            weights_q = weights[1:N:2]
            weights_new = weights_p + weights_q
            weights_ratio = weights_q / weights_new
            
            vs = self.space.metric.log(rec_q, rec_p)
            expand_dims = (1,) * (vs.ndim - 1)
            weights_ratio = weights_ratio.reshape((N//2, *expand_dims))
            rec_new = self.space.metric.exp(weights_ratio * vs, rec_p)
            
            if N == 2:
                self.estimate_ = rec_new[0]
                break
            
            m_rec = rec_new
            weights = weights_new
        
        return self

def std_trimmed(x, axis=1, percent = 5):
    low = np.percentile(x, percent, axis=axis, keepdims=True)
    high = np.percentile(x, 100-percent, axis=axis, keepdims=True)

    x_trimmed = np.where((x >= low) & (x <= high), x, np.nan)
    return np.nanstd(x_trimmed, axis=axis)/np.sqrt(x_trimmed.shape[0])

def mean_trimmed(x, axis=1, percent = 5):
    low = np.percentile(x, percent, axis=axis, keepdims=True)
    high = np.percentile(x, 100-percent, axis=axis, keepdims=True)

    x_trimmed = np.where((x >= low) & (x <= high), x, np.nan)
    return np.nanmean(x_trimmed, axis=axis)