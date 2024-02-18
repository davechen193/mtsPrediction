import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import strptime
from scipy.linalg import qr, solve_triangular
import scipy

def onehot(v, n):
    return np.eye(n)[v,:]

def r2score(true, ests):
    ssres = sum((np.array(true) - np.array(ests))**2)
    sstot = sum((np.array(true) - np.mean(true))**2)
    R2 = 1 - ssres / sstot
    return R2

def integrator(f0, der, dt):
    vals = [f0]
    f = f0
    for d in der[:-1]:
        f += d * dt
        vals.append(f)
    return np.array(vals)

# find the sorted order of the given sequence
def orderOf(arr):
    n = arr.shape[0]
    indicesOrder = np.argsort(arr)
    indexPairs = list(zip(np.arange(n), indicesOrder))
    indexPairs.sort(key=lambda p: p[1])
    valOrder = np.array(list(map(lambda p: p[0], indexPairs)))
    return valOrder

def standard_error_hl(s, window):
    s = pd.Series(s)
    s_mean = np.array([np.mean(s[j-window+1:j+1]) for j in range(s.shape[0])])
    s_std = np.array([np.std(s[j-window+1:j+1]) for j in range(s.shape[0])])
    low = s_mean - s_std * 2
    high = s_mean + s_std * 2
    return low, high

def standardize(ts):
    return (ts - np.mean(ts)) / (2 * np.std(ts))

def weighted_pearson_corr(x, y, w):
    x_mean = np.mean(x); y_mean = np.mean(y)
    numerator = np.sum(w ** 2 * (x - x_mean)*(y - y_mean)) 
    denominator = np.sqrt(np.sum((w * (x - x_mean))**2)*np.sum((w*(y-y_mean))**2))
    return numerator / denominator

def weighted_std(values, weights):
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return np.sqrt(variance)

def data_clip(s, window):
    s = pd.Series(s).rolling(window).apply(lambda s: \
        np.max(s[~np.isinf(s)]) if s.iloc[-1] > np.max(s[~np.isinf(s)]) else s.iloc[-1]
    )
    s = pd.Series(s).rolling(window).apply(lambda s: \
        np.min(s[~np.isinf(s)]) if s.iloc[-1] < np.min(s[~np.isinf(s)]) else s.iloc[-1]
    )
    return s

def least_squares(X, Y, weights=None, params={'lambd': 0}):
    lambd = params['lambd']
    if weights is None:
        weights = np.ones(X.shape[0])
    W = np.diag(weights)

    # multiply each row of X by the corresponding weight
    Xweighted = np.array([weights[i] * X[i,:] for i in range(X.shape[0])])
    return np.linalg.solve(Xweighted.T.dot(X) + lambd * np.identity(X.shape[1]), Xweighted.T.dot(Y))

def least_squares_with_svd(X,Y):
    U, s, Vh = np.linalg.svd(X, full_matrices=False)
    r = len(s)
    Ainv = np.sum([1 / s[i] * Vh.T[:,i].reshape(-1, 1) @ U[:,i].T.reshape(-1, 1).T for i in range(r)], 0)
    return Ainv @ Y