import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage

def get_distance_matrix(corr):
    """Calculate the distance matrix from the correlation matrix."""
    return np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))

def get_quasi_diag(link):
    """Sort clustered items by distance to build the quasi-diagonalized matrix."""
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0]).sort_index()
        
    return sort_ix.tolist()

def get_cluster_var(cov, c_items):
    """Calculate cluster variance using Inverse Variance allocation."""
    cov_ = cov.loc[c_items, c_items]
    w_ = 1.0 / np.diag(cov_)
    w_ /= w_.sum()
    return np.dot(np.dot(w_, cov_), w_)

def get_rec_bipart(cov, sort_ix):
    """Compute HRP allocations recursively down the tree hierarchy."""
    w = pd.Series(1.0, index=sort_ix)
    c_items = [sort_ix]
    
    while len(c_items) > 0:
        c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
        for i in range(0, len(c_items), 2):
            c_items0 = c_items[i]
            c_items1 = c_items[i + 1]
            c_var0 = get_cluster_var(cov, c_items0)
            c_var1 = get_cluster_var(cov, c_items1)
            alpha = 1 - c_var0 / (c_var0 + c_var1)
            w[c_items0] *= alpha
            w[c_items1] *= 1 - alpha
    return w