import numpy as np
import pandas as pd
from scipy.optimize import minimize
from numpy.linalg import inv

def implied_equilibrium_returns(cov_matrix, market_weights=None, risk_aversion=3.0):
    """
    Calculate market equilibrium implied returns via reverse optimization:
    Pi = delta * Sigma * w_mkt
    
    Parameters
    ----------
    cov_matrix : pd.DataFrame
        Asset covariance matrix (N x N).
    market_weights : pd.Series, optional
        Market portfolio weights. Default is equal weighting (1/N).
    risk_aversion : float
        Market risk aversion coefficient (delta). Default is 3.0.
        
    Returns
    -------
    pi : pd.Series
        Implied equilibrium return vector (N).
    """
    n = len(cov_matrix)
    if market_weights is None:
        w_mkt = pd.Series(1.0 / n, index=cov_matrix.index)
    else:
        w_mkt = market_weights.reindex(cov_matrix.index).fillna(1.0 / n)
        w_mkt /= w_mkt.sum()
        
    pi_values = risk_aversion * np.dot(cov_matrix.values, w_mkt.values)
    return pd.Series(pi_values, index=cov_matrix.index)

def default_omega(P, cov_matrix, tau=0.05):
    """
    Calculate default Idzorek view error covariance matrix Omega:
    Omega = diag( P * (tau * Sigma) * P^T )
    
    Parameters
    ----------
    P : np.ndarray or pd.DataFrame
        Pick matrix (K x N).
    cov_matrix : pd.DataFrame
        Asset covariance matrix (N x N).
    tau : float
        Scalar parameter reflecting uncertainty of prior.
        
    Returns
    -------
    omega : np.ndarray
        Diagonal confidence matrix (K x K).
    """
    P_mat = P.values if isinstance(P, (pd.DataFrame, pd.Series)) else np.asarray(P)
    cov_mat = cov_matrix.values if isinstance(cov_matrix, pd.DataFrame) else np.asarray(cov_matrix)
    
    omega = np.diag(np.diag(P_mat.dot(tau * cov_mat).dot(P_mat.T)))
    return omega

def black_litterman_posterior(cov_matrix, pi, P, Q, omega=None, tau=0.05):
    """
    Calculate Black-Litterman posterior returns and posterior covariance matrix.
    
    E[R] = [(tau * Sigma)^-1 + P.T * Omega^-1 * P]^-1 * [(tau * Sigma)^-1 * Pi + P.T * Omega^-1 * Q]
    Sigma_BL = Sigma + [(tau * Sigma)^-1 + P.T * Omega^-1 * P]^-1
    
    Parameters
    ----------
    cov_matrix : pd.DataFrame
        Asset equilibrium covariance matrix (N x N).
    pi : pd.Series
        Market implied equilibrium returns (N).
    P : pd.DataFrame or np.ndarray
        View picking matrix (K x N).
    Q : pd.Series or np.ndarray
        Views expected return vector (K).
    omega : np.ndarray, optional
        Views uncertainty covariance matrix (K x K). Default uses Idzorek formulation.
    tau : float
        Scalar weight indicating uncertainty of market prior (typically 0.01 to 0.1).
        
    Returns
    -------
    posterior_returns : pd.Series
        Black-Litterman posterior expected returns (N).
    posterior_cov : pd.DataFrame
        Black-Litterman posterior covariance matrix (N x N).
    """
    assets = cov_matrix.index
    Sigma = cov_matrix.values
    Pi_vec = pi.values if isinstance(pi, pd.Series) else np.asarray(pi)
    P_mat = P.values if isinstance(P, (pd.DataFrame, pd.Series)) else np.asarray(P)
    Q_vec = Q.values if isinstance(Q, pd.Series) else np.asarray(Q)
    
    if omega is None:
        omega = default_omega(P_mat, cov_matrix, tau=tau)
        
    tau_Sigma = tau * Sigma
    tau_Sigma_inv = inv(tau_Sigma)
    omega_inv = inv(omega)
    
    # Calculate M^-1 = [(tau * Sigma)^-1 + P.T * Omega^-1 * P]^-1
    M_inv = inv(tau_Sigma_inv + P_mat.T.dot(omega_inv).dot(P_mat))
    
    # Posterior expected returns
    rhs = tau_Sigma_inv.dot(Pi_vec) + P_mat.T.dot(omega_inv).dot(Q_vec)
    mu_bl = M_inv.dot(rhs)
    
    # Posterior covariance matrix
    Sigma_bl = Sigma + M_inv
    
    posterior_returns = pd.Series(mu_bl, index=assets)
    posterior_cov = pd.DataFrame(Sigma_bl, index=assets, columns=assets)
    
    return posterior_returns, posterior_cov

def optimize_black_litterman(posterior_returns, posterior_cov, risk_aversion=3.0, bounds=(0.0, 1.0)):
    """
    Compute optimal portfolio weights under Black-Litterman posterior distribution:
    Maximize: w^T * mu_BL - (delta / 2) * w^T * Sigma_BL * w
    Subject to: sum(w) = 1, bounds <= w <= 1
    
    Parameters
    ----------
    posterior_returns : pd.Series
        Black-Litterman posterior returns.
    posterior_cov : pd.DataFrame
        Black-Litterman posterior covariance matrix.
    risk_aversion : float
        Risk aversion parameter.
    bounds : tuple
        Min and max weight bounds per asset.
        
    Returns
    -------
    bl_weights : pd.Series
        Optimal portfolio weights.
    """
    n = len(posterior_returns)
    init_w = np.ones(n) / n
    bnds = tuple(bounds for _ in range(n))
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    
    def objective(w):
        port_ret = np.dot(w, posterior_returns.values)
        port_var = np.dot(w.T, np.dot(posterior_cov.values, w))
        # Negative utility for minimization
        return -(port_ret - 0.5 * risk_aversion * port_var)
        
    res = minimize(objective, init_w, method='SLSQP', bounds=bnds, constraints=constraints)
    return pd.Series(res.x, index=posterior_returns.index)