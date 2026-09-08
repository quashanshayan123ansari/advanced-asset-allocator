import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from scipy.optimize import minimize

from models.hrp import get_distance_matrix, get_quasi_diag, get_rec_bipart
from models.barra_factor import calculate_factor_exposures, build_risk_model
from models.black_litterman import black_litterman_posterior

def fetch_universe_data(tickers, start='2020-01-01', end='2024-01-01'):
    data = yf.download(tickers, start=start, end=end)['Close']
    if isinstance(data, pd.Series):
        data = data.to_frame()
    returns = data.pct_change().dropna()
    return returns

def compute_markowitz_min_variance(cov):
    n = len(cov)
    init_w = np.ones(n) / n
    bounds = tuple((0, 1) for _ in range(n))
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    obj = lambda w: np.dot(w.T, np.dot(cov, w))
    res = minimize(obj, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
    return pd.Series(res.x, index=cov.index)

def plot_dendrogram_and_heatmap(corr, link, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    dendrogram(link, labels=corr.columns.tolist(), ax=ax1, leaf_rotation=90)
    ax1.set_title("Asset Hierarchical Tree Clustering")
    
    sns.heatmap(corr, ax=ax2, cmap="coolwarm", annot=True, fmt=".2f", cbar=True)
    ax2.set_title("Asset Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hrp_clustering.png"), dpi=300)
    plt.close()

def plot_allocation_comparison(hrp_w, markowitz_w, output_dir):
    df = pd.DataFrame({'Hierarchical Risk Parity': hrp_w, 'Markowitz Min-Variance': markowitz_w}) * 100
    ax = df.plot(kind='bar', figsize=(11, 5), color=['#1f77b4', '#aec7e8'], edgecolor='black')
    plt.title("Portfolio Allocation Breakdown: HRP vs. Classical Mean-Variance")
    plt.ylabel("Allocation Weight (%)")
    plt.xlabel("Assets")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "allocation_comparison.png"), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Advanced Portfolio Optimization Engine")
    parser.add_argument('--tickers', nargs='+', default=['SPY', 'QQQ', 'TLT', 'GLD', 'XOM', 'NVDA', 'JPM', 'VNQ'],
                        help="Asset ticker universe")
    parser.add_argument('--start', type=str, default='2020-01-01')
    parser.add_argument('--end', type=str, default='2024-01-01')
    args = parser.parse_args()

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Fetching return series for {len(args.tickers)} assets: {', '.join(args.tickers)}")
    returns = fetch_universe_data(args.tickers, args.start, args.end)
    cov = returns.cov() * 252
    corr = returns.corr()

    # --- HRP Execution ---
    dist_matrix = get_distance_matrix(corr)
    condensed_dist = squareform(dist_matrix, checks=False)
    link = linkage(condensed_dist, method='single')
    sort_ix = get_quasi_diag(link)
    sorted_tickers = corr.columns[sort_ix].tolist()
    hrp_weights = get_rec_bipart(cov, sorted_tickers).loc[args.tickers]

    # --- Benchmark Comparison ---
    markowitz_weights = compute_markowitz_min_variance(cov).loc[args.tickers]

    print("\n" + "="*50)
    print("PORTFOLIO WEIGHT COMPARISON (%)")
    print("="*50)
    comp_table = pd.DataFrame({
        'HRP (%)': (hrp_weights * 100).round(2),
        'Markowitz Min-Var (%)': (markowitz_weights * 100).round(2)
    })
    print(comp_table)

    # --- Export Figures ---
    plot_dendrogram_and_heatmap(corr, link, output_dir)
    plot_allocation_comparison(hrp_weights, markowitz_weights, output_dir)
    print(f"\n[+] Visualizations successfully saved to ./{output_dir}/")

if __name__ == "__main__":
    main()