import numpy as np
import pandas as pd

def construct_style_factors(returns):
    """
    Construct Barra-style equity factors from price/return series:
    - Market (MKT): Equal-weighted market portfolio return
    - Momentum (MOM): Trailing cumulative performance factor
    - Low Volatility (LOWVOL): Inverse volatility factor
    - Quality (QUAL): Risk-adjusted performance factor
    
    Parameters
    ----------
    returns : pd.DataFrame
        Asset daily return DataFrame (T x N).
        
    Returns
    -------
    factors : pd.DataFrame
        Style factor daily returns DataFrame (T x K).
    """
    # 1. Market Factor
    mkt = returns.mean(axis=1)
    
    # 2. Low Volatility Factor: Portfolio long low-vol assets, short high-vol assets
    vols = returns.std(axis=0)
    median_vol = vols.median()
    low_vol_assets = vols[vols <= median_vol].index
    high_vol_assets = vols[vols > median_vol].index
    lowvol = returns[low_vol_assets].mean(axis=1) - returns[high_vol_assets].mean(axis=1)
    
    # 3. Momentum Factor: Portfolio long top trailing returns, short bottom
    cum_returns = (1 + returns).prod() - 1
    median_mom = cum_returns.median()
    high_mom_assets = cum_returns[cum_returns >= median_mom].index
    low_mom_assets = cum_returns[cum_returns < median_mom].index
    mom = returns[high_mom_assets].mean(axis=1) - returns[low_mom_assets].mean(axis=1)
    
    # 4. Quality Factor: Risk-adjusted return proxy
    sharpe = returns.mean(axis=0) / (returns.std(axis=0) + 1e-8)
    median_sharpe = sharpe.median()
    high_qual_assets = sharpe[sharpe >= median_sharpe].index
    low_qual_assets = sharpe[sharpe < median_sharpe].index
    qual = returns[high_qual_assets].mean(axis=1) - returns[low_qual_assets].mean(axis=1)

    factors = pd.DataFrame({
        'Market': mkt,
        'Momentum': mom,
        'Low_Volatility': lowvol,
        'Quality': qual
    }, index=returns.index)
    
    return factors

def calculate_factor_exposures(returns, factors):
    """
    Perform asset-by-asset factor regression to estimate Barra factor exposures.
    Uses Moore-Penrose pseudo-inverse for robust linear algebra without rank deficiency.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Asset daily returns (T x N).
    factors : pd.DataFrame
        Factor daily returns (T x K).
        
    Returns
    -------
    X : pd.DataFrame
        Factor exposures matrix (N x K).
    factor_cov : pd.DataFrame
        Annualized factor covariance matrix (K x K).
    specific_var : pd.Series
        Annualized specific (idiosyncratic) variance per asset (N).
    residuals : pd.DataFrame
        Asset residuals (T x N).
    """
    assets = returns.columns
    factor_names = factors.columns
    
    # Standardize factors cross-time (mean 0, std 1) for clean factor beta interpretation
    factors_std = (factors - factors.mean()) / (factors.std() + 1e-8)
    
    F_mat = factors_std.values # T x K
    F_pinv = np.linalg.pinv(F_mat) # K x T
    
    betas = []
    specific_variances = {}
    residuals_dict = {}
    
    for asset in assets:
        y = returns[asset].values
        beta_i = F_pinv.dot(y) # K factor loadings
        resid_i = y - F_mat.dot(beta_i)
        
        betas.append(beta_i)
        residuals_dict[asset] = resid_i
        specific_variances[asset] = np.var(resid_i) * 252.0
        
    X = pd.DataFrame(betas, index=assets, columns=factor_names)
    factor_cov = factors_std.cov() * 252.0
    specific_var = pd.Series(specific_variances)
    residuals = pd.DataFrame(residuals_dict, index=returns.index)
    
    return X, factor_cov, specific_var, residuals

def build_risk_model(X, factor_cov, specific_var):
    """
    Reconstruct the full Barra covariance matrix:
    V = X * F * X^T + Delta
    
    Parameters
    ----------
    X : pd.DataFrame
        Factor exposures matrix (N x K).
    factor_cov : pd.DataFrame
        Factor covariance matrix (K x K).
    specific_var : pd.Series
        Specific variance vector (N).
        
    Returns
    -------
    V : pd.DataFrame
        Reconstructed asset covariance matrix (N x N).
    """
    systematic_cov = X.dot(factor_cov).dot(X.T)
    delta = np.diag(specific_var.values)
    V_values = systematic_cov.values + delta
    
    return pd.DataFrame(V_values, index=X.index, columns=X.index)

def decompose_portfolio_risk(weights, X, factor_cov, specific_var):
    """
    Decompose total portfolio variance into systematic (factor) and specific components.
    
    Parameters
    ----------
    weights : pd.Series
        Portfolio asset weights (N).
    X : pd.DataFrame
        Factor exposures matrix (N x K).
    factor_cov : pd.DataFrame
        Factor covariance matrix (K x K).
    specific_var : pd.Series
        Specific variance vector (N).
        
    Returns
    -------
    risk_summary : dict
        Dictionary containing total_variance, systematic_variance, specific_variance,
        pct_systematic, pct_specific, and factor_risk_contributions.
    """
    w = weights.reindex(X.index).values
    
    # Systematic Risk: w^T * X * F * X^T * w
    exp_p = X.T.dot(w) # Portfolio factor exposure (K)
    systematic_var = float(exp_p.T.dot(factor_cov).dot(exp_p))
    
    # Specific Risk: w^T * Delta * w
    specific_var_total = float(np.sum((w ** 2) * specific_var.values))
    
    total_var = systematic_var + specific_var_total
    
    pct_sys = (systematic_var / total_var) * 100.0 if total_var > 0 else 0.0
    pct_spec = (specific_var_total / total_var) * 100.0 if total_var > 0 else 0.0
    
    # Factor breakdown
    factor_contributions = exp_p * factor_cov.dot(exp_p) / total_var if total_var > 0 else exp_p * 0.0
    
    return {
        'total_variance': total_var,
        'total_volatility': np.sqrt(total_var),
        'systematic_variance': systematic_var,
        'specific_variance': specific_var_total,
        'pct_systematic': pct_sys,
        'pct_specific': pct_spec,
        'portfolio_factor_exposures': exp_p,
        'factor_contributions': factor_contributions
    }