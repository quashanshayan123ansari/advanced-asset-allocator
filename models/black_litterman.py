import numpy as np
from numpy.linalg import inv

def black_litterman_posterior(tau, cov_matrix, pi, P, Q, omega):
    """
    Calculate the posterior expected returns using Black-Litterman.
    E[R] = [(tau * Sigma)^-1 + P.T * Omega^-1 * P]^-1 * [(tau * Sigma)^-1 * Pi + P.T * Omega^-1 * Q]
    """
    # tau * Sigma
    ts = tau * cov_matrix
    ts_inv = inv(ts)
    
    # P.T * Omega^-1 * P
    omega_inv = inv(omega)
    term2 = P.T.dot(omega_inv).dot(P)
    
    # Calculate Posterior Expected Returns
    part1 = inv(ts_inv + term2)
    part2 = ts_inv.dot(pi) + P.T.dot(omega_inv).dot(Q)
    
    posterior_returns = part1.dot(part2)
    return posterior_returns