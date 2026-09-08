import numpy as np
import statsmodels.api as sm

def calculate_factor_exposures(returns, factors):
    """Cross-sectional regression to calculate factor exposures (X)."""
    X = sm.add_constant(factors)
    model = sm.OLS(returns, X).fit()
    return model.params, model.resid

def build_risk_model(X, factor_cov, specific_var):
    """Build the risk model: V = X * F * X.T + Delta"""
    systematic_risk = X.dot(factor_cov).dot(X.T)
    idiosyncratic_risk = np.diag(specific_var)
    return systematic_risk + idiosyncratic_risk