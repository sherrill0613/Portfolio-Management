import pandas as pd
import numpy as np
import cvxopt as opt
from cvxopt import blas, solvers
import seaborn as sns
from matplotlib import pyplot as plt


# import the data
pos = pd.read_csv('positions.csv')
ret = pd.read_csv('Returns.csv')
index = pd.read_csv('index_df.csv')
data_full = pd.read_csv('CryptoDataOHLCV_483.csv')
pos = pos.iloc[1600:, :]
ret = ret.iloc[1600:, :]
close_pivot = data_full.pivot_table(values='close', index='date', columns='symbol')
return_full = close_pivot.pct_change().apply(lambda x: np.log(1 + x))
index = index.iloc[:, 1:]


def get_mu_cov(pos_array):
    ticker = np.array(pos_array)
    mu = return_full[ticker].mean()
    cov_mat = return_full[ticker].cov()
    return mu * 365, cov_mat * 365


def optimal_portfolio(mu, cov_mat):
    n = 50
    N = 100
    mus = [10 ** (5.0 * t / N - 1.0) for t in range(N)]
    # Convert to cvxopt matrices
    S = opt.matrix(cov_mat.values)
    pbar = opt.matrix(mu.values)

    # Create constraint matrices
    G = -opt.matrix(np.eye(n))  # negative n x n identity matrix
    h = opt.matrix(0.0, (n, 1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu * S, -pbar, G, h, A, b, kktsolver='ldl', options={'kktreg': 1e-9})['x'] for mu in mus]
    # CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(np.abs(blas.dot(x, S * x))) for x in portfolios]
    # CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(np.abs(m1[2] / m1[0]))
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b, kktsolver='ldl', options={'kktreg': 1e-9})['x']
    return pd.Series(wt)


# mean variance optimization
weights_df = pd.DataFrame(columns=range(50))
for i in range(0, len(index), 7):
    mu, cov_mat = get_mu_cov(index.iloc[i, :])
    weights = optimal_portfolio(mu=mu.T, cov_mat=cov_mat)
    weights_df = weights_df.append(weights, ignore_index=True)
weights_df = weights_df.loc[weights_df.index.repeat(7)]

# plot Cov
sns.heatmap(cov_mat)
plt.show()

# Calculate cumulative return
ret_index = pd.DataFrame(index=range(1003), columns=range(50))
for row in range(len(index)):
    for item in range(len(index.columns)):
        ret_index.iloc[row, item] = return_full[index.iloc[row, item]][1600 + row]


weights_df = weights_df.reset_index()
weights_df.drop(['index'], axis=1, inplace=True)
port_return = pd.DataFrame(index=range(1003), columns=range(50))
for row in range(len(port_return)):
    for item in range(len(port_return.columns)):
        port_return.iloc[row, item] = weights_df.iloc[row, item] * ret_index.iloc[row, item]
port_return['port return'] = port_return.sum(axis=1)
port_return = port_return.fillna(0)
port_return['Cumulative'] = port_return['port return'] + 1
port_return['Cumulative'] = port_return['Cumulative'].cumprod()
plt.figure(figsize=(15, 9), dpi=80)
sns.lineplot(x=port_return['Cumulative'].index, y=port_return['Cumulative'].values)
plt.ylabel('Portfolio Cumulative Return')
plt.show()

Sharpe_Ratio = port_return['port return'].mean() / port_return['port return'].std() * np.sqrt(365)
CAGR = (port_return['Cumulative'] ** 1/1003 + 1) ** 365 - 1