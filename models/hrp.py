import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

def get_distance_matrix(corr):
    """
    Calculate correlation distance matrix following Marcos López de Prado (2016).
    
    d_{i,j} = sqrt(0.5 * (1 - corr_{i,j}))
    
    Parameters
    ----------
    corr : pd.DataFrame
        Asset correlation matrix.
        
    Returns
    -------
    dist : pd.DataFrame
        Distance matrix.
    """
    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))
    return dist

def get_quasi_diag(link):
    """
    Reorder linkage matrix items to form a quasi-diagonal block matrix.
    
    Parameters
    ----------
    link : np.ndarray
        Linkage matrix output from scipy.cluster.hierarchy.linkage.
        
    Returns
    -------
    sort_ix : list
        Reordered index positions of assets.
    """
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
    """
    Calculate sub-cluster variance using inverse-variance weights allocation.
    
    Parameters
    ----------
    cov : pd.DataFrame
        Asset covariance matrix.
    c_items : list
        List of asset labels in the cluster.
        
    Returns
    -------
    c_var : float
        Variance of the cluster portfolio.
    """
    cov_ = cov.loc[c_items, c_items]
    w_ = 1.0 / np.diag(cov_)
    w_ /= w_.sum()
    return np.dot(np.dot(w_, cov_), w_)

def get_rec_bipart(cov, sort_ix):
    """
    Compute HRP allocation weights recursively down the hierarchical dendrogram.
    
    Parameters
    ----------
    cov : pd.DataFrame
        Asset covariance matrix.
    sort_ix : list
        Quasi-diagonalized list of asset names.
        
    Returns
    -------
    w : pd.Series
        Hierarchical Risk Parity portfolio weights.
    """
    w = pd.Series(1.0, index=sort_ix)
    c_items = [sort_ix]
    
    while len(c_items) > 0:
        c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
        for i in range(0, len(c_items), 2):
            c_items0 = c_items[i]
            c_items1 = c_items[i + 1]
            c_var0 = get_cluster_var(cov, c_items0)
            c_var1 = get_cluster_var(cov, c_items1)
            alpha = 1.0 - c_var0 / (c_var0 + c_var1)
            w[c_items0] *= alpha
            w[c_items1] *= (1.0 - alpha)
    return w

def run_hrp_pipeline(cov, corr, method='single'):
    """
    High-level execution function for Hierarchical Risk Parity.
    
    Parameters
    ----------
    cov : pd.DataFrame
        Asset covariance matrix.
    corr : pd.DataFrame
        Asset correlation matrix.
    method : str
        Linkage clustering method ('single', 'complete', 'ward', 'average').
        
    Returns
    -------
    hrp_weights : pd.Series
        Optimal HRP weights indexed by asset ticker.
    link : np.ndarray
        Linkage matrix for dendrogram visualization.
    sorted_tickers : list
        Sorted ticker names from quasi-diagonalization.
    """
    dist_matrix = get_distance_matrix(corr)
    condensed_dist = squareform(dist_matrix.values, checks=False)
    link = linkage(condensed_dist, method=method)
    sort_ix = get_quasi_diag(link)
    sorted_tickers = corr.columns[sort_ix].tolist()
    hrp_weights = get_rec_bipart(cov, sorted_tickers).loc[cov.index]
    return hrp_weights, link, sorted_tickers