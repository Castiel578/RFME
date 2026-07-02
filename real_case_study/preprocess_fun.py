import numpy as np
import math
from geomstats.geometry.pre_shape import PreShapeSpace

def inner_f(X1, X2):
    return np.trace(X1 @ X2.T)

def norm_f(X):
    return np.trace((X @ X.T)) ** 0.5


def geo_sp(t, X1, X2):
    if inner_f(X1, X2)>=1:
        return None
    else:
        theta = np.arccos(np.min([inner_f(X1, X2), 1]))
        p = 1/np.sin(theta)*(np.sin((1-t)*theta)*X1 + np.sin(t*theta)*X2)
        return p

def exp_sp(X1, v):
    norm_v = norm_f(v)
    if norm_v == 0:
        q = X1
    else:
        q = np.cos(norm_v) * X1 + np.sin(norm_v) * v / norm_v
    return q

def log_sp(X1, X2):
    inner = inner_f(X1, X2)
    if inner>1: inner = 1
    if inner<-1: inner = -1
    if inner != 1:
        theta = np.arccos(inner)
        v = theta / np.sin(theta) * (X2 - X1 * np.cos(theta))
    else:
        v = 0 * (X2 - X1)
    return v

def obj_center(X):
    colmean = np.mean(X, axis=0)
    return X - colmean

def obj_scale(X):
    norm_X = norm_f(X)
    return X/norm_X

def rotate(X1, X2):
    X = X1.T @ X2
    U, S, Vt = np.linalg.svd(X , full_matrices=False)
    if np.linalg.det(X) > 0:
        return U @ Vt
    else:
        id = np.eye(X1.shape[1])
        id[-1, -1] = -1
        return U @ id @ Vt

def obj_rotate(X1, X2):
    O = rotate(X2, X1)
    #return (O @ X1.T).T X1 @ O.T
    return X1 @ O.T

def obj_align(Xs, ind = 0):
    n = Xs.shape[0]
    g1 = Xs[ind][:-1]
    for j in range(n):
        if j == ind:
            continue
        #g2 = obj_rotate(Xs[j], Xs[ind])[:-1]
        g2 = Xs[j][:-1]
        dist = np.zeros(g1.shape[0])
        for i in range(g1.shape[0]):
            perm_g2 = np.vstack([g2[i:], g2[:i]])
            dist[i] = norm_f(log_sp(g1, obj_rotate(perm_g2, g1)))
        arg_min = np.argmin(dist)
        permed_g2 = np.vstack([g2[arg_min:], g2[: arg_min]])
        permed_g2 = np.vstack([permed_g2, permed_g2[0]])
        Xs[j] = obj_rotate(permed_g2, Xs[ind])
    return(Xs)

def unify(Xs):
    Xs = Xs - np.mean(Xs, axis = 1, keepdims = True)
    dims = Xs.shape

    for i in range(dims[0]):
        Xs[i] = obj_scale(Xs[i])
        #Xs[i] = obj_rotate(Xs[i], rotate(Xs[0], Xs[i]))
    return Xs

def SSE(mean, data):
    n = data.shape[0]
    sse = 0

    for i in range(n):
        d = norm_f(log_sp(mean, obj_rotate(data[i], mean)))**2
        sse = sse + d

    return sse

def Rec_Frechet_mean_geom(Xs):
    n, a, b = Xs.shape
    preshape = PreShapeSpace(a, b)
    preshape.equip_with_group_action('rotations')
    preshape.equip_with_quotient()

    weights = np.full(n, 1)
    m_rec = preshape.projection(Xs)

    N = m_rec.shape[0]
    k = N - 2 ** math.floor(math.log(N, 2))
    if k != 0:
        rec_new = np.empty((k, *m_rec.shape[1:]))
        weights_new = np.empty(k)
        for i in range(k):
            v = preshape.quotient.metric.log(m_rec[2*i+1], m_rec[2*i])
            rec_new[i] = preshape.quotient.metric.exp((weights[2*i+1] / (weights[2*i] + weights[2*i+1]))*v, m_rec[2*i] )
            weights_new[i] = weights[2*i] + weights[2*i+1]
        m_rec = np.concatenate((rec_new, m_rec[2*k:]), axis = 0)
        weights = np.concatenate((weights_new, weights[2*k:]), axis = 0)

    while(True):
            N = m_rec.shape[0]
            weights_new = np.empty(N//2)
            rec_new = np.empty((N//2, *m_rec.shape[1:]))
            
            for i in range(N//2):
                #geod_func = preshape.quotient.metric.geodesic(m_rec[2*i], m_rec[2*i+1])
                #rec_new[i] = geod_func(weights[2*i+1] / (weights[2*i] + weights[2*i+1]))[0]
                v = preshape.quotient.metric.log(m_rec[2*i+1], m_rec[2*i])
                rec_new[i] = preshape.quotient.metric.exp((weights[2*i+1] / (weights[2*i] + weights[2*i+1]))*v, m_rec[2*i])
                weights_new[i] = weights[2*i] + weights[2*i+1]
            
            if N == 2:
                mean = rec_new[0]
                break
            
            m_rec = rec_new
            weights = weights_new 
    
    return mean
    