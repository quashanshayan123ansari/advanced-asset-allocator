# Advanced Convex Optimization & Factor Risk Modeling

A production-grade quantitative finance repository addressing empirical covariance noise, ill-conditioned matrices, and non-normal asset distributions using **Hierarchical Risk Parity (HRP)**, **Barra-style Multi-Factor Risk Decomposition**, and **Black-Litterman Bayesian Allocation**.

---

## Key Architectures

### 1. Hierarchical Risk Parity (HRP)
- **Problem**: Standard Mean-Variance optimization $(\Sigma^{-1})$ breaks under ill-conditioned or non-positive definite empirical covariance matrices.
- **Solution**: Marcos López de Prado’s machine learning graph-clustering approach:
  1. **Tree Clustering**: Computes correlation distance $d_{i,j} = \sqrt{\frac{1}{2}(1 - \rho_{i,j})}$ and builds a single-linkage dendrogram.
  2. **Quasi-Diagonalization**: Reorders asset covariance to cluster similar risk drivers together.
  3. **Recursive Bisection**: Allocates inverse-variance risk top-down without matrix inversion.

### 2. Multi-Factor Risk Model with Barra-Style Decomposition
- **Style Factors**: Market (MKT), Momentum (MOM), Low Volatility (LOWVOL), Quality (QUAL).
- **Time-Series Regression**: Extracts exposure matrix $X$ ($N \times K$), factor covariance $F$ ($K \times K$), and specific variances $\Delta$ ($N \times N$).
- **Variance Decomposition**:
  $$V = X F X^T + \Delta$$
  Decomposes total portfolio risk into **Systematic Factor Risk** vs **Idiosyncratic Specific Risk**.

### 3. Black-Litterman Model with Bayesian Priors
- **Market Equilibrium Prior**: Reverse optimization to calculate implied equilibrium returns:
  $$\Pi = \delta \Sigma w_{\text{mkt}}$$
- **Subjective Investor Views**: Combines market equilibrium priors $\Pi$ with absolute/relative investor views $(P, Q, \Omega)$.
- **Bayesian Posterior Update**:
  $$\mu_{BL} = \left[(\tau \Sigma)^{-1} + P^T \Omega^{-1} P\right]^{-1} \left[(\tau \Sigma)^{-1} \Pi + P^T \Omega^{-1} Q\right]$$
  $$\Sigma_{BL} = \Sigma + \left[(\tau \Sigma)^{-1} + P^T \Omega^{-1} P\right]^{-1}$$

---

## Project Structure

```text
advanced-risk-models/
├── models/
│   ├── __init__.py
│   ├── hrp.py             # Hierarchical clustering & recursive bisection engine
│   ├── barra_factor.py    # Style factor extraction, exposures & risk decomposition
│   └── black_litterman.py # Prior reverse optimization, views & posterior solver
├── output/                # Generated visualizations & charts
│   ├── hrp_clustering.png
│   ├── allocation_comparison.png
│   ├── barra_factor_exposures.png
│   └── black_litterman_returns.png
├── utils/                 # Matrix helper functions
├── main.py                # End-to-end unified pipeline runner
├── requirements.txt       # Project dependencies
└── README.md
```

---

## Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/quashanshayan123ansari/Blackliterrman.git
   cd advanced-risk-models
   ```

2. **Set up Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Engine

To run the complete pipeline with default tickers (`SPY`, `QQQ`, `TLT`, `GLD`, `XOM`, `NVDA`, `JPM`, `VNQ`):

```bash
python main.py
```

To run with a custom universe of tickers and date range:

```bash
python main.py --tickers AAPL MSFT GOOGL AMZN TSLA --start 2021-01-01 --end 2024-06-01
```

---

## Outputs & Visualizations

Running `main.py` generates 4 high-resolution charts in `./output/`:

1. **`hrp_clustering.png`**: Asset dendrogram tree-clustering and correlation matrix heatmap.
2. **`allocation_comparison.png`**: Weight breakdown across HRP, Markowitz Minimum Variance, and Black-Litterman.
3. **`barra_factor_exposures.png`**: Barra Style Factor Exposure heatmap ($X$) and Systematic vs. Specific variance pie chart.
4. **`black_litterman_returns.png`**: Implied Equilibrium Priors ($\Pi$) vs. Black-Litterman Posterior Expected Returns ($\mu_{BL}$).