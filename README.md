# Advanced Convex Optimization & Factor Risk Modeling

A production-grade Python implementation addressing the mathematical instabilities of Markowitz Mean-Variance frameworks on noisy, high-dimensional empirical covariance matrices.

## Key Architectures

* **Hierarchical Risk Parity (HRP)**: Implements Marcos López de Prado’s machine learning tree-clustering approach. Computes distance matrices $D_{i,j} = \sqrt{\frac{1}{2}(1 - \rho_{i,j})}$, quasi-diagonalizes the dendrogram, and recursively allocates inverse-variance risk without matrix inversion $(\Sigma^{-1})$.
* **Barra-Style Multi-Factor Risk Decomposition**: Decomposes asset variance into systematic factor risk and idiosyncratic specific risk:
  $$V = XFX^T + \Delta$$
* **Bayesian Black-Litterman Allocation**: Blends reverse-optimized market equilibrium priors $(\Pi = \lambda \Sigma w_{mkt})$ with subjective investor views $(P, Q, \Omega)$ to produce stable posterior return distributions.

## Project Structure

```text
advanced-risk-models/
├── data/                  # Empirical series cache
├── models/
│   ├── __init__.py
│   ├── hrp.py             # Tree-clustering & recursive bisection logic
│   ├── barra_factor.py    # Cross-sectional factor regression & risk engine
│   └── black_litterman.py # Prior-posterior Bayesian blending
├── output/                # Saved figures (dendrograms, frontiers)
├── utils/                 # Matrix transformations & metrics
├── .gitignore
├── main.py                # Generalized pipeline execution
├── README.md
└── requirements.txt