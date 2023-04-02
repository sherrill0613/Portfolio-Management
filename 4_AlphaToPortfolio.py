
import pandas as pd
import numpy as np
import datetime


from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib import rc

import seaborn as sns

import pickle

from scipy.io import savemat


sns.set_style('white')
rc('mathtext', default='regular')

with open('Alpha_all.pickle', 'rb') as f:
    Alpha_all = pickle.load(f)

Return_all = Alpha_all[list(Alpha_all.keys())[5]]
Volume_all = Alpha_all[list(Alpha_all.keys())[4]]

Volume_all.fillna(0, inplace = True)
Return_all.fillna(0, inplace = True)

TraditionMacroFactors = pd.read_csv("TraditionMacroFactors.csv",index_col = 0)

namelist = Volume_all.columns.to_list()
alphalist = list(Alpha_all.keys())[6:]
timeline = Return_all.index.to_list()

# Search for alphas
rank_IR_dataset = pd.DataFrame(index=alphalist, columns=[str(i + 1) for i in range(14)])
T, N = Return_all.shape
for day in range(14):
    days = day + 1

    rank_alpha = []
    rank_IR = []
    for i in range(46):
        alpha = alphalist[i]

        Alpha_single = Alpha_all[alpha]
        Alpha_single.fillna(0, inplace=True)
        residual_singleAlpha_TN = pd.DataFrame(index=timeline, columns=namelist)
        residual_singleAlpha_TN.iloc[:, :] = 0
        alpha_return_rate_list = []
        for t in range(1600, T - days - 1):
            t0 = timeline[t]
            t1 = timeline[t+1]
            X_k = Alpha_single.iloc[t, :]
            X_EBC = np.repeat(np.array([TraditionMacroFactors.iloc[t, :]]), 483, axis=0)
            # 𝑋_𝑘^𝑡= 𝛽_𝐸 𝑋_𝐸+𝛽_𝐵 𝑋_𝐵+𝛽_𝐶 𝑋_𝐶+ 𝜀_𝑘^𝑡
            OLS = np.linalg.lstsq(np.array(X_EBC), np.array(np.nan_to_num(X_k)), rcond=None)[0]
            residual_singleAlpha = X_k - np.matmul(X_EBC, OLS)
            residual_singleAlpha.fillna(0, inplace=True)
            residual_singleAlpha_standadize = (residual_singleAlpha - np.array(residual_singleAlpha).mean()) / np.array(residual_singleAlpha).std()
            residual_singleAlpha_standadize.fillna(0, inplace=True)
            residual_singleAlpha_TN.loc[t0, :] = residual_singleAlpha_standadize

            t_forward = timeline[t + days]
            R_t_forward = (Return_all.loc[t1:t_forward, :] + 1).cumprod(axis=0).iloc[-1, :]
            X_EBC_alpha = np.column_stack((X_EBC, residual_singleAlpha_standadize))
            # 𝑅_(𝑡+𝑑)= 𝑓_𝐸 𝑋_𝐸+𝑓_𝐵 𝑋_𝐵+𝑓_𝐶 𝑋_𝐶+ 〖𝑓_𝑘 𝜀〗_𝑘^ + 𝜀_(𝑡+𝑑)
            OLS = np.linalg.lstsq(X_EBC_alpha, R_t_forward.T, rcond=None)[0]
            alpha_return_rate = OLS[-1]
            alpha_return_rate_list.append(alpha_return_rate)

        # 𝐸(𝑓_𝑘 )=252∗ (𝑓_𝑘/𝑑)
        annualized_alpha_return_rate = 365 * (np.array(alpha_return_rate_list).mean()) / days
        # IR(𝑓_𝑘 )=√252∗(((𝑓_𝑘/𝑑) ))/(𝜎(𝑓_𝑘/𝑑))
        IR_alpha_return = np.sqrt(365) * (np.array(alpha_return_rate_list).mean() / days) / ((np.array(alpha_return_rate_list) / days).std())

        rank_alpha.append(annualized_alpha_return_rate)
        rank_IR.append(IR_alpha_return)
    rank_IR_dataset.iloc[:, day] = rank_IR

rank_IR_dataset_sorted = rank_IR_dataset.sort_values(by = "7", ascending = False)

plt.figure(figsize = (16,4))
plt.bar(rank_IR_dataset_sorted.columns, rank_IR_dataset_sorted.iloc[0,:])
plt.show()


# alpha analysis
AlphaAll_rate = pd.DataFrame(index=timeline, columns=alphalist)
AlphaAll_rate.iloc[:, :] = 0
real_alpha_return_dataset = pd.DataFrame(index=timeline, columns=namelist)  # 7 days later alpha return
real_alpha_return_dataset.iloc[:, :] = 0
AlphaAllMultiAsset = np.zeros((T, N, 46))
for t in range(1600, T - 8):
    t0 = timeline[t]
    t1 = timeline[t + 1]

    X_EBC = np.repeat(np.array([TraditionMacroFactors.iloc[t, :]]), 483, axis=0)

    for i in range(46):
        alpha = alphalist[i]
        Alpha_single = Alpha_all[alpha]
        Alpha_single.fillna(0, inplace=True)
        X_k = Alpha_single.iloc[t, :]

        OLS = np.linalg.lstsq(np.array(X_EBC), np.array(np.nan_to_num(X_k)), rcond=None)[0]
        residual_singleAlpha = X_k - np.matmul(X_EBC, OLS)  ## 46*1
        residual_singleAlpha = residual_singleAlpha.fillna(0)
        residual_singleAlpha_standadize = (residual_singleAlpha - np.array(residual_singleAlpha).mean()) / np.array(
            residual_singleAlpha).std()
        residual_singleAlpha_standadize = residual_singleAlpha_standadize.fillna(0)
        AlphaAllMultiAsset[t, :, i] = residual_singleAlpha_standadize

    days = 7
    t_forward = timeline[t + days]
    R_t_forward = (Return_all.loc[t1:t_forward, :] + 1).cumprod(axis=0).iloc[-1, :]

    X_EBC_alpha = np.column_stack((X_EBC, AlphaAllMultiAsset[t, :, :]))

    OLS = np.linalg.lstsq(X_EBC_alpha, R_t_forward.T, rcond=None)[0]

    alpha_return_rate_list = OLS[-46:]
    basic_beta = OLS[:6]
    real_alpha_return = R_t_forward - np.matmul(X_EBC, basic_beta)

    AlphaAll_rate.iloc[t, :] = alpha_return_rate_list
    real_alpha_return_dataset.iloc[t, :] = real_alpha_return

alpha_return_forward = pd.DataFrame(index = timeline, columns = namelist) # 7 days later alpha return
alpha_return_forward.iloc[:,:] = 0
for t in range(1600, T-8):
    t0 = timeline[t]
    t1 = timeline[t+1]
    t_forward = timeline[t+6]
    # N*K * K*1
    alpha_return_forward.loc[t0,:] = np.matmul(AlphaAllMultiAsset[t,:,:], (AlphaAll_rate.loc[t0:t_forward, :].sum()/7).T)

real_alpha_return_dataset.fillna(0,inplace = True)
alpha_return_forward.fillna(0,inplace = True)
Corr = []
for t in range(1600, T-8):
    tmp = np.corrcoef(alpha_return_forward.iloc[t,:],real_alpha_return_dataset.iloc[t,:])[0,1]
    Corr.append(tmp)
# all alpha IC
np.array(Corr).mean()


# change all alpha to selected alpha
alpha_7days_IR = rank_IR_dataset_sorted.loc[:, "7"]
compare_IC = []
for x in range(1,9):
    alpha_selected = alpha_7days_IR[
        (alpha_7days_IR <= alpha_7days_IR.quantile(0.1)) | (alpha_7days_IR >= alpha_7days_IR.quantile(0.1 * x))]
    alpha_selected_list = alpha_selected.index.to_list()

    AlphaAll_rate = pd.DataFrame(index=timeline, columns=alpha_selected_list)
    AlphaAll_rate.iloc[:, :] = 0
    real_alpha_return_dataset = pd.DataFrame(index=timeline, columns=namelist)  # 7 days later alpha return
    real_alpha_return_dataset.iloc[:, :] = 0
    AlphaAllMultiAsset = np.zeros((T, N, len(alpha_selected_list)))
    for t in range(1600, T - 8):
        t0 = timeline[t]
        t1 = timeline[t + 1]

        X_EBC = np.repeat(np.array([TraditionMacroFactors.iloc[t, :]]), 483, axis=0)

        for i in range(len(alpha_selected_list)):
            alpha = alpha_selected_list[i]
            Alpha_single = Alpha_all[alpha]

            Alpha_single.fillna(0, inplace=True)
            X_k = Alpha_single.iloc[t, :]

            OLS = np.linalg.lstsq(np.array(X_EBC), np.array(np.nan_to_num(X_k)), rcond=None)[0]
            residual_singleAlpha = X_k - np.matmul(X_EBC, OLS)  ## K*1
            residual_singleAlpha = residual_singleAlpha.fillna(0)
            residual_singleAlpha_standadize = (residual_singleAlpha - np.array(residual_singleAlpha).mean()) / np.array(
                residual_singleAlpha).std()
            residual_singleAlpha_standadize = residual_singleAlpha_standadize.fillna(0)
            AlphaAllMultiAsset[t, :, i] = residual_singleAlpha_standadize

        days = 7
        t_forward = timeline[t + days]
        R_t_forward = (Return_all.loc[t1:t_forward, :] + 1).cumprod(axis=0).iloc[-1, :]
        X_EBC_alpha = np.column_stack((X_EBC, AlphaAllMultiAsset[t, :, :]))

        OLS = np.linalg.lstsq(X_EBC_alpha, R_t_forward.T, rcond=None)[0]

        alpha_return_rate_list = OLS[-len(alpha_selected_list):]
        basic_beta = OLS[:6]
        real_alpha_return = R_t_forward - np.matmul(X_EBC, basic_beta)

        AlphaAll_rate.iloc[t, :] = alpha_return_rate_list
        real_alpha_return_dataset.iloc[t, :] = real_alpha_return

    alpha_return_forward = pd.DataFrame(index=timeline, columns=namelist)  # 7 days later alpha return
    alpha_return_forward.iloc[:, :] = 0
    for t in range(1600, T - 8):
        t0 = timeline[t]
        t1 = timeline[t + 1]
        t_forward = timeline[t + 6]
        # N*K * K*1
        alpha_return_forward.loc[t0, :] = np.matmul(AlphaAllMultiAsset[t, :, :],
                                                    (AlphaAll_rate.loc[t0:t_forward, :].sum() / 7).T)

    real_alpha_return_dataset.fillna(0, inplace=True)
    alpha_return_forward.fillna(0, inplace=True)
    Corr = []
    for t in range(1600, T - 8):
        tmp = np.corrcoef(alpha_return_forward.iloc[t, :], real_alpha_return_dataset.iloc[t, :])[0, 1]
        Corr.append(tmp)

    # compare alpha IC
    compare_IC.append(np.array(Corr).mean())
compare_IC

plt.figure(figsize = (16,8))
compare_IC_dataset = pd.DataFrame(compare_IC, index = [str(int(i+1)*0.1) for i in range(8)])
plt.plot(compare_IC_dataset)


# alpha strategy
alpha_7days_IR = rank_IR_dataset_sorted.loc[:, "7"]
alpha_selected = alpha_7days_IR[(alpha_7days_IR<=alpha_7days_IR.quantile(0.1)) | (alpha_7days_IR>=alpha_7days_IR.quantile(0.1*6))]
alpha_selected_list = alpha_selected.index.to_list()

AlphaAll_rate = pd.DataFrame(index=timeline, columns=alpha_selected_list)
AlphaAll_rate.iloc[:, :] = 0
real_alpha_return_dataset = pd.DataFrame(index=timeline, columns=namelist)  # 7 days later alpha return
real_alpha_return_dataset.iloc[:, :] = 0
AlphaAllMultiAsset = np.zeros((T, N, len(alpha_selected_list)))
for t in range(1600, T - 8):
    t0 = timeline[t]
    t1 = timeline[t + 1]

    X_EBC = np.repeat(np.array([TraditionMacroFactors.iloc[t, :]]), 483, axis=0)

    for i in range(len(alpha_selected_list)):
        alpha = alpha_selected_list[i]
        Alpha_single = Alpha_all[alpha]
        Alpha_single.fillna(0, inplace=True)
        X_k = Alpha_single.iloc[t, :]

        OLS = np.linalg.lstsq(np.array(X_EBC), np.array(np.nan_to_num(X_k)), rcond=None)[0]
        residual_singleAlpha = X_k - np.matmul(X_EBC, OLS)  ## K*1
        residual_singleAlpha = residual_singleAlpha.fillna(0)
        residual_singleAlpha_standadize = (residual_singleAlpha - np.array(residual_singleAlpha).mean()) / np.array(
            residual_singleAlpha).std()
        residual_singleAlpha_standadize = residual_singleAlpha_standadize.fillna(0)
        AlphaAllMultiAsset[t, :, i] = residual_singleAlpha_standadize

    days = 7
    t_forward = timeline[t + days]
    R_t_forward = (Return_all.loc[t1:t_forward, :] + 1).cumprod(axis=0).iloc[-1, :]
    X_EBC_alpha = np.column_stack((X_EBC, AlphaAllMultiAsset[t, :, :]))

    OLS = np.linalg.lstsq(X_EBC_alpha, R_t_forward.T, rcond=None)[0]

    alpha_return_rate_list = OLS[-len(alpha_selected_list):]
    basic_beta = OLS[:6]
    real_alpha_return = R_t_forward - np.matmul(X_EBC, basic_beta)

    AlphaAll_rate.iloc[t, :] = alpha_return_rate_list
    real_alpha_return_dataset.iloc[t, :] = real_alpha_return

alpha_return_forward = pd.DataFrame(index = timeline, columns = namelist) # 7 days later alpha return
alpha_return_forward.iloc[:,:] = 0
for t in range(1600, T-8):
    t0 = timeline[t]
    t1 = timeline[t+1]
    t_forward = timeline[t+6]
    # N*K * K*1
    alpha_return_forward.loc[t0,:] = np.matmul(AlphaAllMultiAsset[t,:,:], (AlphaAll_rate.loc[t0:t_forward, :].sum()/7).T)

real_alpha_return_dataset.fillna(0,inplace = True)
alpha_return_forward.fillna(0,inplace = True)


positions = pd.DataFrame(index=timeline, columns=namelist)
positions.iloc[:, :] = 0
returns = positions.copy()
volumes = positions.copy()
alphas = positions.copy()
index_df = pd.DataFrame(index=timeline[1600:2594], columns = [str(i+1) for i in range(50)])
for t in range(142):
    t0_index = timeline[t*7+1600]
    t6_index = timeline[t*7+1606]
    t0 = t*7+1600
    Top50Index = alpha_return_forward.iloc[t0, :].rank(axis=0).sort_values().iloc[:50].index
    index_df.loc[t0_index:t6_index, :] = Top50Index
    positions.loc[t0_index:t6_index, Top50Index] = 1
    # returns.loc[t0,:] = Return_all.iloc[t,:]
    # volumes.loc[t0,:] = Volume_all.iloc[t,:]
    # alphas.loc[t0,:] = alpha_return_forward.iloc[t,:]


portfolio = positions * Return_all

portfolio1 = pd.DataFrame({"return": portfolio.iloc[1600:2600,:].mean(axis = 1),
                           "BTC-USD": Return_all.iloc[1600:2600,:]["BTC-USD"]})
portfolio1['log_return'] = np.log1p(portfolio1["return"])
portfolio1["BTC-USD_return"] = np.log1p(portfolio1["BTC-USD"])
portfolio1['Portfolio_Performance'] = np.exp(portfolio1['log_return'].cumsum())
portfolio1["BTC_USD_Return"] = np.exp(portfolio1["BTC-USD_return"].cumsum())
portfolio1[['Portfolio_Performance', "BTC_USD_Return"]].plot(figsize = (16,9))

# our portfolio performance
Sharpe_ratio = portfolio1['log_return'].mean()/portfolio1['log_return'].std()*np.sqrt(365)
def Max_drawdown(df):
    return (df.cummax()-df).max()
MaxDrawdown = Max_drawdown(portfolio1['log_return'].cumsum())
Cumulative_Return = portfolio1['Portfolio_Performance'][-1] - 1
print("Sharpe_ratio:",Sharpe_ratio,"\nMaxDrawdown:", MaxDrawdown, "\nCumulative_Return: ",Cumulative_Return)

# the BTC-USD benchmark performance
Sharpe_ratio = portfolio1['BTC-USD_return'].mean()/portfolio1['BTC-USD_return'].std()*np.sqrt(365)
MaxDrawdown = Max_drawdown(portfolio1['BTC-USD_return'].cumsum())
Cumulative_Return = portfolio1['BTC_USD_Return'][-1] - 1
print("Sharpe_ratio:",Sharpe_ratio,"\nMaxDrawdown:", MaxDrawdown, "\nCumulative_Return: ",Cumulative_Return)


positions.to_csv("positions.csv")
portfolio.to_csv("portfolio_return.csv")
index_df.to_csv("index_df.csv")

savemat("Returns.mat", returns)
savemat("Volumes.mat", volumes)
savemat("live_company.mat", positions)
savemat("Factors.mat", alphas)


