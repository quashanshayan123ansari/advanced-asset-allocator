import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from scipy.optimize import minimize

from models.hrp import run_hrp_pipeline
from models.barra_factor import (
    construct_style_factors,
    calculate_factor_exposures,
    build_risk_model,
    decompose_portfolio_risk
)
from models.black_litterman import (
    implied_equilibrium_returns,
    black_litterman_posterior,
    optimize_black_litterman
)

def fetch_universe_data(tickers, start='2020-01-01', end='2024-01-01'):
    """Fetch close prices and calculate daily returns."""
    data = yf.download(tickers, start=start, end=end)['Close']
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.dropna(axis=1, how='all')
    returns = data.pct_change().dropna()
    return returns

def compute_markowitz_min_variance(cov):
    """Compute benchmark Markowitz Minimum Variance portfolio."""
    n = len(cov)
    init_w = np.ones(n) / n
    bounds = tuple((0, 1) for _ in range(n))
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    obj = lambda w: np.dot(w.T, np.dot(cov, w))
    res = minimize(obj, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
    return pd.Series(res.x, index=cov.index)

def plot_dendrogram_and_heatmap(corr, link, output_dir):
    """Plot asset dendrogram tree-clustering and correlation matrix heatmap."""
    from scipy.cluster.hierarchy import dendrogram
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    dendrogram(link, labels=corr.columns.tolist(), ax=ax1, leaf_rotation=90)
    ax1.set_title("HRP Asset Hierarchical Tree Clustering", fontsize=12, fontweight='bold')
    
    sns.heatmap(corr, ax=ax2, cmap="coolwarm", annot=True, fmt=".2f", cbar=True, vmin=-1, vmax=1)
    ax2.set_title("Asset Correlation Matrix", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hrp_clustering.png"), dpi=300)
    plt.close()

def plot_allocation_comparison(hrp_w, markowitz_w, bl_w, output_dir):
    """Plot portfolio allocation breakdown across HRP, Markowitz, and Black-Litterman."""
    df = pd.DataFrame({
        'Hierarchical Risk Parity (HRP)': hrp_w,
        'Markowitz Min-Variance': markowitz_w,
        'Black-Litterman Posterior': bl_w
    }) * 100.0
    
    ax = df.plot(kind='bar', figsize=(12, 6), color=['#1f77b4', '#aec7e8', '#2ca02c'], edgecolor='black', width=0.8)
    plt.title("Portfolio Allocation Weight Comparison (%)", fontsize=14, fontweight='bold')
    plt.ylabel("Allocation Weight (%)", fontsize=12)
    plt.xlabel("Asset Tickers", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "allocation_comparison.png"), dpi=300)
    plt.close()

def plot_barra_factor_exposures(X, risk_decomp, output_dir):
    """Plot Barra Style Factor exposures heatmap and portfolio risk contribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Heatmap of Factor Exposures X
    sns.heatmap(X, ax=ax1, cmap="viridis", annot=True, fmt=".2f", cbar=True)
    ax1.set_title("Barra Style Factor Exposures (X Matrix)", fontsize=12, fontweight='bold')
    
    # Pie chart of Systematic vs Specific Risk
    labels = [f"Systematic Risk\n({risk_decomp['pct_systematic']:.1f}%)", 
              f"Specific Risk\n({risk_decomp['pct_specific']:.1f}%)"]
    sizes = [risk_decomp['pct_systematic'], risk_decomp['pct_specific']]
    colors = ['#4c72b0', '#55a868']
    
    ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, 
            wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
    ax2.set_title("Portfolio Risk Variance Decomposition", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "barra_factor_exposures.png"), dpi=300)
    plt.close()

def plot_black_litterman_returns(prior_pi, posterior_mu, output_dir):
    """Plot Market Implied Priors vs Black-Litterman Posterior Expected Returns."""
    df = pd.DataFrame({
        'Implied Equilibrium Prior (Pi)': prior_pi * 100.0,
        'Black-Litterman Posterior (mu_BL)': posterior_mu * 100.0
    })
    
    ax = df.plot(kind='bar', figsize=(11, 5), color=['#ff7f0e', '#1f77b4'], edgecolor='black')
    plt.title("Black-Litterman Return Distribution: Market Prior vs. Subjective Posterior", fontsize=13, fontweight='bold')
    plt.ylabel("Annualized Expected Return (%)", fontsize=11)
    plt.xlabel("Assets", fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "black_litterman_returns.png"), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Advanced Portfolio Optimization & Factor Risk Engine")
    parser.add_argument('--tickers', nargs='+', default=['SPY', 'QQQ', 'TLT', 'GLD', 'XOM', 'NVDA', 'JPM', 'VNQ'],
                        help="Asset ticker universe")
    parser.add_argument('--start', type=str, default='2020-01-01')
    parser.add_argument('--end', type=str, default='2024-01-01')
    args = parser.parse_args()

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(" ADVANCED CONVEX OPTIMIZATION & FACTOR RISK ENGINE")
    print("=" * 70)
    print(f"Fetching empirical series for {len(args.tickers)} assets: {', '.join(args.tickers)}")
    
    returns = fetch_universe_data(args.tickers, args.start, args.end)
    tickers = returns.columns.tolist()
    cov = returns.cov() * 252.0
    corr = returns.corr()

    # ---------------------------------------------------------
    # 1. HRP (Hierarchical Risk Parity) Execution
    # ---------------------------------------------------------
    print("\n[1/3] Executing Marcos Lopez de Prado's Hierarchical Risk Parity (HRP)...")
    hrp_weights, link, sorted_tickers = run_hrp_pipeline(cov, corr)
    markowitz_weights = compute_markowitz_min_variance(cov)

    # ---------------------------------------------------------
    # 2. Barra-Style Multi-Factor Risk Decomposition
    # ---------------------------------------------------------
    print("[2/3] Constructing Barra Style Factors & Multi-Factor Risk Decomposition...")
    factors = construct_style_factors(returns)
    X, factor_cov, specific_var, residuals = calculate_factor_exposures(returns, factors)
    barra_cov = build_risk_model(X, factor_cov, specific_var)
    risk_decomp = decompose_portfolio_risk(hrp_weights, X, factor_cov, specific_var)

    # ---------------------------------------------------------
    # 3. Black-Litterman Model with Bayesian Priors
    # ---------------------------------------------------------
    print("[3/3] Blending Market Equilibrium Priors with Subjective Views (Black-Litterman)...")
    delta = 3.0
    mkt_weights = pd.Series(1.0 / len(tickers), index=tickers)
    pi = implied_equilibrium_returns(cov, mkt_weights, risk_aversion=delta)

    # Define Subjective Investor Views (P and Q matrices):
    # View 1: NVDA outperforming SPY by +6.0% (0.06)
    # View 2: TLT outperforming GLD by +3.0% (0.03)
    P = pd.DataFrame(0.0, index=['NVDA > SPY', 'TLT > GLD'], columns=tickers)
    P.loc['NVDA > SPY', 'NVDA'] = 1.0
    P.loc['NVDA > SPY', 'SPY'] = -1.0
    
    P.loc['TLT > GLD', 'TLT'] = 1.0
    P.loc['TLT > GLD', 'GLD'] = -1.0
    
    Q = pd.Series([0.06, 0.03], index=['NVDA > SPY', 'TLT > GLD'])

    posterior_mu, posterior_cov = black_litterman_posterior(cov, pi, P, Q, tau=0.05)
    bl_weights = optimize_black_litterman(posterior_mu, posterior_cov, risk_aversion=delta)

    # ---------------------------------------------------------
    # CLI Reporting Tables
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print(" PORTFOLIO ALLOCATION SUMMARY MATRIX (%)")
    print("=" * 70)
    summary_df = pd.DataFrame({
        'HRP (%)': (hrp_weights * 100.0).round(2),
        'Markowitz Min-Var (%)': (markowitz_weights * 100.0).round(2),
        'Black-Litterman (%)': (bl_weights * 100.0).round(2),
        'Equilibrium Prior (Pi) (%)': (pi * 100.0).round(2),
        'BL Posterior (mu) (%)': (posterior_mu * 100.0).round(2)
    })
    print(summary_df.to_string())

    print("\n" + "=" * 70)
    print(" BARRA MULTI-FACTOR EXPOSURE MATRIX (X)")
    print("=" * 70)
    print(X.round(3).to_string())

    print("\n" + "=" * 70)
    print(" BARRA PORTFOLIO RISK DECOMPOSITION")
    print("=" * 70)
    print(f" Total Portfolio Volatility:  {risk_decomp['total_volatility']*100:.2f}%")
    print(f" Systematic (Factor) Risk:   {risk_decomp['pct_systematic']:.2f}%")
    print(f" Specific (Idiosyncratic):  {risk_decomp['pct_specific']:.2f}%")

    # ---------------------------------------------------------
    # Visualizations Export
    # ---------------------------------------------------------
    plot_dendrogram_and_heatmap(corr, link, output_dir)
    plot_allocation_comparison(hrp_weights, markowitz_weights, bl_weights, output_dir)
    plot_barra_factor_exposures(X, risk_decomp, output_dir)
    plot_black_litterman_returns(pi, posterior_mu, output_dir)

    print("\n" + "=" * 70)
    print(f"[+] All model visualizations successfully generated & saved to ./{output_dir}/:")
    print("    - ./output/hrp_clustering.png")
    print("    - ./output/allocation_comparison.png")
    print("    - ./output/barra_factor_exposures.png")
    print("    - ./output/black_litterman_returns.png")
    print("=" * 70)

if __name__ == "__main__":
    main()