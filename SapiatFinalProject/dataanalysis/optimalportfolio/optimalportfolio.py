import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import time
from datetime import timedelta
import math
import glob
import prettytable
import cvxopt as opt
from cvxopt import blas, solvers
from pathlib import Path
def loadassetfile():
    rootd = Path(__file__).parent.parent.parent
    rootd = rootd.joinpath(rootd, "inputdata")
    bond = pd.read_csv(str(rootd.joinpath(rootd, "bond.csv").absolute()), parse_dates=True)
    bond.rename(columns={'bond_ret': 'bond_daily_ret'}, inplace=True)
    bond['date'] = pd.to_datetime(bond.date)
    equity = pd.read_csv(r'inputdata\equity.csv', parse_dates=True)
    equity['equity_daily_ret'] = equity.adj_close.pct_change()
    equity['date'] = pd.to_datetime(equity.date)
    forex = pd.read_csv(r'inputdata\forex.csv', parse_dates=True)
    forex['forex_daily_ret'] = forex.adj_close.pct_change()
    forex['date'] = pd.to_datetime(forex.date)
    gc = pd.read_csv(r'inputdata\gc.csv', parse_dates=True)
    gc['gold_daily_ret'] = gc.adj_close.pct_change()
    gc['date'] = pd.to_datetime(gc.date)
    return (bond, equity, forex, gc)
def getreturns(bond, equity, forex, gc):
    combined = pd.concat(
        [equity.set_index('date'), bond.set_index('date'), gc.set_index('date'), forex.set_index('date') \
         ], axis=1, join='inner')
    returns = combined.loc[:, ['equity_daily_ret', 'bond_daily_ret', 'gold_daily_ret', 'forex_daily_ret']]
    returns.reset_index(inplace=True)
    returns['year'] = returns['date'].dt.year
    returns['month'] = returns['date'].dt.month
    returns['quarter'] = returns.month.apply(lambda v: 'Q1' if v <= 3 else 'Q2' if v <= 6 else 'Q3' if v <= 9 else 'Q4')
    returns['year_quarter'] = returns.year.astype(str) + '-' + returns.quarter.astype(str)
    returns.set_index('date', inplace=True)
    returns = returns.fillna('0')
    return (returns)
def readstage1data():
    rootd = Path(__file__).parent.parent.parent
    rootd = rootd.joinpath(rootd, "outputdata")
    rootd = rootd.joinpath(rootd, "Stage1")
    rootd = rootd.joinpath(rootd, "*.csv")
    filenametab = []
    for filename in glob.glob(str(rootd.absolute())):
        filenametab.append(filename)
    filenametab.sort(reverse=True)
    data = pd.read_csv(filenametab[0], parse_dates=True)
    data['date']= pd.to_datetime(data.date)
    data.set_index('date', inplace=True)
    return data
def readstage1dataLR():
    rootd = Path(__file__).parent.parent.parent
    rootd = rootd.joinpath(rootd, "outputdata")
    rootd = rootd.joinpath(rootd, "Stage1")
    rootd = rootd.joinpath(rootd, "LR")
    rootd = rootd.joinpath(rootd, "*.csv")
    filenametab = []
    for filename in glob.glob(str(rootd.absolute())):
        filenametab.append(filename)
    filenametab.sort(reverse=True)
    data = pd.read_csv(filenametab[0], parse_dates=True)
    data['date']= pd.to_datetime(data.date)
    data['date']= pd.to_datetime(data.date)
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['quarter'] = data.month.apply(
        lambda v: 'Q1' if v <= 3 else 'Q2' if v <= 6 else 'Q3' if v <= 9 else 'Q4')
    data['year_quarter'] = data.year.astype(str) + '-' + data.quarter.astype(str)
    data.set_index('date', inplace=True)
    return data
# covariance matrix from mgarch model
def get_covar_matrix_hat(x):
    X = np.array([x['equity_var_hat'], x['bond_equity_cov_hat'], x['equity_gold_cov_hat'], x['equity_forex_cov_hat'],
                  x['bond_equity_cov_hat'], x['bond_var_hat'], x['bond_gold_cov_hat'], x['bond_forex_cov_hat'],
                  x['equity_gold_cov_hat'], x['bond_gold_cov_hat'], x['gold_var_hat'], x['forex_gold_cov_hat'],
                  x['equity_forex_cov_hat'], x['bond_forex_cov_hat'], x['forex_gold_cov_hat'], x['forex_var_hat']])
    X = X.astype(np.float)
    return opt.matrix(X, (4, 4))
def rand_weights(n):
    ''' Produces n random weights that sum to 1 '''
    np.seed(1000)
    k = np.random.rand(n)
    return k / sum(k)
def random_portfolio(returns, return_vec, cov):
    '''
    Returns the mean and standard deviation of returns for a random portfolio
    '''
    p = np.asmatrix(returns)
    w = np.asmatrix(rand_weights(return_vec.shape[1]))
    C = np.asmatrix(cov)
    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)
    # This recursion reduces outliers to keep plots pretty
    if sigma > 2:
        return random_portfolio(returns)
    return mu, sigma
def optimal_portfolio(returns, S):
    n = len(returns)
    N = 100
    mus = [10 ** (5.0 * t / N - 1.0) for t in range(N)]
    # Convert to cvxopt matrices
    pbar = opt.matrix(returns)
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))  # negative n x n identity matrix
    h = opt.matrix(0.0, (n, 1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    # Calculate efficient frontier weights using quadratic programming
    solvers.options['show_progress'] = False
    portfolios = [solvers.qp(mu * S, -pbar, G, h, A, b)['x']
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = []
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = []
    risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]
    risks = [(sum(risks) / len(risks)) if math.isnan(x) else x for x in risks]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(abs(m1[2] / m1[0]))
    # CALCULATE THE OPTIMAL PORTFOLIO
    solvers.options['show_progress'] = False
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks, portfolios
def find_ret(RISK, RETURN, tgt_risk):
    '''
    RISK is a np.array() of risk levels
    RETURN is a np.array() of associated optimum returns
    '''
    assert len(RISK) == len(RETURN)
    last = None
    for (risk, ret) in zip(RISK, RETURN):
        if risk > tgt_risk:
            return last
        last = ret
    return last

def buildweightshat(inputstage1, df_daily_ret_act,lookback):
    mean_lookback_returns = []
    quarterly_cov_hat = []
    df_daily_ret_act.reset_index(inplace=True)
    inputstage1.reset_index(inplace=True)
    for x in inputstage1.to_dict('records'):
        sub = df_daily_ret_act[df_daily_ret_act['date'] < x['date']].tail(lookback)
        return_vec = sub[['equity_daily_ret', 'bond_daily_ret', 'gold_daily_ret', 'forex_daily_ret']].mean().values
        sigma = get_covar_matrix_hat(x)
        mean_lookback_returns += [return_vec]
        quarterly_cov_hat += [sigma]
    risks = []
    weights = []
    for ret, cov in zip(mean_lookback_returns, quarterly_cov_hat):
        w, re, ri, p = optimal_portfolio(ret, cov)
        risks += [ri]
        weights += [p]
    table = {}
    weights_table = []
    for i in range(len(inputstage1)):
        table = {'risks': risks[i], 'weights': weights[i]}
        table = pd.DataFrame(table)
        table = table.sort_values(by=['risks'])
        # low_w = find_weights(table.risks, table.weights, None, 0.01)
        # med_w = find_weights(table.risks, table.weights,0.01, 0.025)
        # high_w = find_weights(table.risks, table.weights, 0.04, None)
        low_w = find_weights(table.risks, table.weights, 0.003)
        med_w = find_weights(table.risks, table.weights, 0.005)
        high_w = find_weights(table.risks, table.weights,0.007)
        # low_w = find_weights(table.risks, table.weights, table.risks.quantile(q=0.1))
        # med_w = find_weights(table.risks, table.weights, table.risks.quantile(q=0.5))
        # high_w = find_weights(table.risks, table.weights,table.risks.quantile(q=0.9))
        g = {'equity_low_w_hat': float(low_w[0]),
             'bond_low_w_hat': float(low_w[1]),
             'gold_low_w_hat': float(low_w[2]),
             'forex_low_w_hat': float(low_w[3]),
             'equity_med_w_hat': float(med_w[0]),
             'bond_med_w_hat': float(med_w[1]),
             'gold_med_w_hat': float(med_w[2]),
             'forex_med_w_hat': float(med_w[3]),
             'equity_high_w_hat': float(high_w[0]),
             'bond_high_w_hat': float(high_w[1]),
             'gold_high_w_hat': float(high_w[2]),
             'forex_high_w_hat': float(high_w[3])}
        weights_table += [g]
    weights_table = pd.DataFrame(weights_table)
    weights_table['calculation_date'] = inputstage1['date'] + timedelta(days=1)
    weights_table['year'] = weights_table['calculation_date'].dt.year
    weights_table['month'] = weights_table['calculation_date'].dt.month
    weights_table['quarter'] = weights_table.month.apply(
        lambda v: 'Q1' if v <= 3 else 'Q2' if v <= 6 else 'Q3' if v <= 9 else 'Q4')
    weights_table['year_quarter'] = weights_table.year.astype(str) + '-' + weights_table.quarter.astype(str)
    # weights_table = weights_table.drop(['year','month','quarter'], axis = 0)
    weights_table.set_index('year_quarter', inplace=True)
    weights_table = weights_table.fillna('0')
    return weights_table
def buildweightshis(inputstage1, df_daily_ret_act,lookback):
    mean_lookback_returns_his = []
    quarterly_cov_hat_his = []
    df_daily_ret_act.reset_index(inplace=True)
    inputstage1.reset_index(inplace=True)
    for x in inputstage1.to_dict('records'):
        sub = df_daily_ret_act[df_daily_ret_act['date'] < x['date']].tail(lookback)
        return_vec = sub[['equity_daily_ret', 'bond_daily_ret', 'gold_daily_ret', 'forex_daily_ret']].mean().values
        sub =  sub.drop(['date','index'], axis=1)
        sub = sub.astype(np.float)
        return_vec_cov = sub.cov()
        sigma = opt.matrix(return_vec_cov.values)
        mean_lookback_returns_his += [return_vec]
        quarterly_cov_hat_his += [sigma]
    risks_his = []
    weights_his = []
    for ret1, cov1 in zip(mean_lookback_returns_his, quarterly_cov_hat_his):
        w, re, ri1, p1 = optimal_portfolio(ret1, cov1)
        risks_his += [ri1]
        weights_his += [p1]
    table = {}
    weights_table_his = []
    for i in range(len(inputstage1)):
        table = {'risks': risks_his[i], 'weights': weights_his[i]}
        table = pd.DataFrame(table)
        table = table.sort_values(by=['risks'])
        # low_w = find_weights(table.risks, table.weights,None, 0.01)
        # med_w = find_weights(table.risks, table.weights,0.01, 0.025)
        # high_w = find_weights(table.risks, table.weights,0.04, None)
        low_w = find_weights(table.risks, table.weights, 0.003)
        med_w = find_weights(table.risks, table.weights, 0.005)
        high_w = find_weights(table.risks, table.weights,0.007)
        # low_w = find_weights(table.risks, table.weights, table.risks.quantile(q=0.1))
        # med_w = find_weights(table.risks, table.weights, table.risks.quantile(q=0.5))
        # high_w = find_weights(table.risks, table.weights,table.risks.quantile(q=0.9))
        g = {'equity_low_w_his': float(low_w[0]),
             'bond_low_w_his': float(low_w[1]),
             'gold_low_w_his': float(low_w[2]),
             'forex_low_w_his': float(low_w[3]),
             'equity_med_w_his': float(med_w[0]),
             'bond_med_w_his': float(med_w[1]),
             'gold_med_w_his': float(med_w[2]),
             'forex_med_w_his': float(med_w[3]),
             'equity_high_w_his': float(high_w[0]),
             'bond_high_w_his': float(high_w[1]),
             'gold_high_w_his': float(high_w[2]),
             'forex_high_w_his': float(high_w[3])}
        weights_table_his += [g]
    weights_table_his = pd.DataFrame(weights_table_his)
    weights_table_his['calculation_date'] = inputstage1['date'] + timedelta(days=1)
    weights_table_his['year'] = weights_table_his['calculation_date'].dt.year
    weights_table_his['month'] = weights_table_his['calculation_date'].dt.month
    weights_table_his['quarter'] = weights_table_his.month.apply(
        lambda v: 'Q1' if v <= 3 else 'Q2' if v <= 6 else 'Q3' if v <= 9 else 'Q4')
    weights_table_his['year_quarter'] = weights_table_his.year.astype(str) + '-' + weights_table_his.quarter.astype(str)
    weights_table_his.set_index('year_quarter', inplace=True)
    weights_table_his = weights_table_his.fillna('0')
    return weights_table_his

def buildweightsLR(inputstage1LR, df_daily_ret_act,lookback):
    mean_lookback_returns_lr = []
    quarterly_cov_hat_lr = []
    df_daily_ret_act.reset_index(inplace=True)
    inputstage1LR.reset_index(inplace=True)
    for x in inputstage1LR.to_dict('records'):
        sub = df_daily_ret_act[df_daily_ret_act['date'] < x['date']].tail(lookback)
        return_vec = sub[['equity_daily_ret', 'bond_daily_ret', 'gold_daily_ret', 'forex_daily_ret']].mean().values
        sub =  sub.drop(['date','index'], axis=1)
        sigma = get_covar_matrix_hat(x)
        mean_lookback_returns_lr += [return_vec]
        quarterly_cov_hat_lr += [sigma]
    risks_lr = []
    weights_lr = []
    for ret1, cov1 in zip(mean_lookback_returns_lr, quarterly_cov_hat_lr):
        w, re, ri1, p1 = optimal_portfolio(ret1, cov1)
        risks_lr += [ri1]
        weights_lr += [p1]
    table = {}
    weights_table_lr = []
    for i in range(len(inputstage1LR)):
        table = {'risks': risks_lr[i], 'weights': weights_lr[i]}
        table = pd.DataFrame(table)
        table = table.sort_values(by=['risks'])
        # low_w = find_weights(table.risks, table.weights,None, 0.01)
        # med_w = find_weights(table.risks, table.weights,0.01, 0.025)
        # high_w = find_weights(table.risks, table.weights,0.04, None)
        low_w = find_weights(table.risks, table.weights, 0.003)
        med_w = find_weights(table.risks, table.weights, 0.005)
        high_w = find_weights(table.risks, table.weights,0.007)
        # low_w = find_weights(table.risks, table.weights, table.risks.quantile(q=0.1))
        # med_w = find_weights(table.risks, table.weights, table.risks.quantile(q=0.5))
        # high_w = find_weights(table.risks, table.weights,table.risks.quantile(q=0.9))
        g = {'equity_low_w_lr': float(low_w[0]),
             'bond_low_w_lr': float(low_w[1]),
             'gold_low_w_lr': float(low_w[2]),
             'forex_low_w_lr': float(low_w[3]),
             'equity_med_w_lr': float(med_w[0]),
             'bond_med_w_lr': float(med_w[1]),
             'gold_med_w_lr': float(med_w[2]),
             'forex_med_w_lr': float(med_w[3]),
             'equity_high_w_lr': float(high_w[0]),
             'bond_high_w_lr': float(high_w[1]),
             'gold_high_w_lr': float(high_w[2]),
             'forex_high_w_lr': float(high_w[3])}
        weights_table_lr += [g]
    weights_table_lr = pd.DataFrame(weights_table_lr)
    weights_table_lr['calculation_date'] = inputstage1LR['date'] + timedelta(days=1)
    weights_table_lr['year'] = weights_table_lr['calculation_date'].dt.year
    weights_table_lr['month'] = weights_table_lr['calculation_date'].dt.month
    weights_table_lr['quarter'] = weights_table_lr.month.apply(
        lambda v: 'Q1' if v <= 3 else 'Q2' if v <= 6 else 'Q3' if v <= 9 else 'Q4')
    weights_table_lr['year_quarter'] = weights_table_lr.year.astype(str) + '-' + weights_table_lr.quarter.astype(str)
    weights_table_lr.set_index('year_quarter', inplace=True)
    weights_table_lr = weights_table_lr.fillna('0')
    return weights_table_lr


def dailyportfolioreturn(returns_act, inputstage1_act, weights_table_hat, weights_table_his,weights_table_LR):
    start_date = '1990-03-31'
    end_date = '2020-03-31'
    returns_act = returns_act[start_date:end_date]  # Add one more qtr end here
    weights_table_his = weights_table_his.drop(['year', 'month', 'quarter'], axis=1)
    weights_table_hat = weights_table_hat.drop(['year', 'month', 'quarter', 'calculation_date'], axis=1)
    weights_table_LR  = weights_table_LR.drop(['year', 'month', 'quarter'], axis=1)

    final_csv_temp = pd.merge(left=returns_act.reset_index(), right=weights_table_his, left_on='year_quarter',
                              right_on='year_quarter').set_index(['date'])

    final_csv_temp_mgarch = pd.merge(left=final_csv_temp.reset_index(), right=weights_table_hat, left_on='year_quarter',
                                right_on='year_quarter').set_index(['date'])

    final_csv_output = pd.merge(left=final_csv_temp_mgarch.reset_index(), right=weights_table_LR, left_on='year_quarter',
                              right_on='year_quarter').set_index(['date'])

    final_csv_output = final_csv_output.drop(['year', 'month', 'quarter', 'year_quarter'], axis=1)

    return_arr = np.array(
        final_csv_output[['equity_daily_ret', 'bond_daily_ret', 'gold_daily_ret', 'forex_daily_ret']].T, dtype=float)

    final_csv_output['ret_low_his'] = np.einsum("ij,ij->j", return_arr
                                                , np.array(
            final_csv_output[['equity_low_w_his', 'bond_low_w_his', 'gold_low_w_his', 'forex_low_w_his']].T,
            dtype=float))

    final_csv_output['ret_med_his'] = np.einsum("ij,ij->j", return_arr
                                                , np.array(
            final_csv_output[['equity_med_w_his', 'bond_med_w_his', 'gold_med_w_his', 'forex_med_w_his']].T,
            dtype=float))
    final_csv_output['ret_high_his'] = np.einsum("ij,ij->j", return_arr
                                                 , np.array(
            final_csv_output[['equity_high_w_his', 'bond_high_w_his', 'gold_high_w_his', 'forex_high_w_his']].T,
            dtype=float))

    final_csv_output['ret_low_hat'] = np.einsum("ij,ij->j", return_arr
                                                , np.array(
            final_csv_output[['equity_low_w_hat', 'bond_low_w_hat', 'gold_low_w_hat', 'forex_low_w_hat']].T,
            dtype=float))
    final_csv_output['ret_med_hat'] = np.einsum("ij,ij->j", return_arr
                                                , np.array(
            final_csv_output[['equity_med_w_hat', 'bond_med_w_hat', 'gold_med_w_hat', 'forex_med_w_hat']].T,
            dtype=float))
    final_csv_output['ret_high_hat'] = np.einsum("ij,ij->j", return_arr
                                                 , np.array(
            final_csv_output[['equity_high_w_hat', 'bond_high_w_hat', 'gold_high_w_hat', 'forex_high_w_hat']].T,
            dtype=float))

    final_csv_output['ret_low_lr'] = np.einsum("ij,ij->j", return_arr
                                                , np.array(
            final_csv_output[['equity_low_w_lr', 'bond_low_w_lr', 'gold_low_w_lr', 'forex_low_w_lr']].T,
            dtype=float))
    final_csv_output['ret_med_lr'] = np.einsum("ij,ij->j", return_arr
                                                , np.array(
            final_csv_output[['equity_med_w_lr', 'bond_med_w_lr', 'gold_med_w_lr', 'forex_med_w_lr']].T,
            dtype=float))
    final_csv_output['ret_high_lr'] = np.einsum("ij,ij->j", return_arr
                                                 , np.array(
            final_csv_output[['equity_high_w_lr', 'bond_high_w_lr', 'gold_high_w_lr', 'forex_high_w_lr']].T,
            dtype=float))

    return (final_csv_output)
def savecsvfile(csvstructure, filenamemask):
    if filenamemask.startswith('Stage2OutputReturns'):
        filename = r'outputdata\Stage2\Returns\SapiatStage2Out' + filenamemask + time.strftime("%Y%m%d%H%M%S") + '.csv'
    else:
        filename = r'outputdata\Stage2\Weights\SapiatStage2Out' +  filenamemask + time.strftime("%Y%m%d%H%M%S") + '.csv'
    csvstructure.to_csv(filename)
    print(f' Processing Complete, File :- {filename} has been created\r')
def apply_cumsum(returns_series):
    initial_investment = 100.00
    cumsumret = (returns_series.add(1).cumprod()) * initial_investment
    cumsumret.iat[0] = initial_investment
    return cumsumret


def main():
    bond, equity,forex,gc = loadassetfile()
    inputstage1 = readstage1data()
    inputstage1LR = readstage1dataLR()
    inputstage1_act = inputstage1.copy()
    inputstage1LR_act=inputstage1LR.copy()
    returns = getreturns(bond, equity, forex, gc)
    df_daily_ret_act = returns.copy()
    df_daily_ret_act = df_daily_ret_act.drop(['year', 'month', 'quarter','year_quarter'], axis=1)
    returns_act = returns.copy()
    weights_table_hat = buildweightshat(inputstage1, df_daily_ret_act,504)
    weights_table_his = buildweightshis(inputstage1, df_daily_ret_act,504)
    weights_table_LR = buildweightsLR(inputstage1LR, df_daily_ret_act,504)

    dailyportfolioreturncsv = dailyportfolioreturn(returns_act, inputstage1_act, weights_table_hat, weights_table_his,weights_table_LR)

    # dailyportfolioreturncsv['ret_low_his_cum'] = apply_cumsum(dailyportfolioreturncsv.ret_low_his)
    # dailyportfolioreturncsv['ret_low_hat_cum'] = apply_cumsum(dailyportfolioreturncsv.ret_low_hat)
    # dailyportfolioreturncsv['ret_low_lr_cum'] = apply_cumsum(dailyportfolioreturncsv.ret_low_lr)
    #
    # dailyportfolioreturncsv['ret_med_his_cum'] = apply_cumsum(dailyportfolioreturncsv.ret_med_his)
    # dailyportfolioreturncsv['ret_med_hat_cum'] = apply_cumsum(dailyportfolioreturncsv.ret_med_hat)
    # dailyportfolioreturncsv['ret_med_lr_cum'] = apply_cumsum(dailyportfolioreturncsv.ret_med_lr)
    #
    # dailyportfolioreturncsv['ret_high_his_cum'] = apply_cumsum(dailyportfolioreturncsv.ret_high_his)
    # dailyportfolioreturncsv['ret_high_hat_cum'] = apply_cumsum(dailyportfolioreturncsv.ret_high_hat)
    # dailyportfolioreturncsv['ret_high_lr_cum'] = apply_cumsum(dailyportfolioreturncsv.ret_high_lr)
    savecsvfile(dailyportfolioreturncsv, 'Stage2OutputReturns')
    # savecsvfile(weights_table_hat, 'Stage2OutputWeightsHat')
    # savecsvfile(weights_table_his, 'Stage2OutputWeightsHis')
    return dailyportfolioreturncsv
if __name__ == '__main__':
    dailyportfolioreturncsv = main()
