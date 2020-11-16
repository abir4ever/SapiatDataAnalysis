import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import time
import math
import prettytable
import cvxopt as opt
from cvxopt import blas, solvers
from os import listdir
from os.path import isfile, join
from pathlib import Path
def loadassetfile():
    rootd = Path(__file__).parent.parent.parent
    rootd = rootd.joinpath(rootd, "inputdata")
    bond = pd.read_csv(str(rootd.joinpath(rootd, "bond.csv").absolute()), parse_dates=True)
    bond['date'] = pd.to_datetime(bond.date)
    equity = pd.read_csv(r'inputdata\equity.csv', parse_dates=True)
    equity['equity_ret'] = equity.adj_close.pct_change()
    equity['date'] = pd.to_datetime(equity.date)
    forex = pd.read_csv(r'inputdata\forex.csv', parse_dates=True)
    forex['forex_ret'] = forex.adj_close.pct_change()
    forex['date'] = pd.to_datetime(forex.date)
    gc = pd.read_csv(r'inputdata\gc.csv', parse_dates=True)
    gc['gc_ret'] = gc.adj_close.pct_change()
    gc['date'] = pd.to_datetime(gc.date)
    return (bond, equity, forex, gc)
def getreturns(bond, equity, forex, gc):
    combined = pd.concat(
        [equity.set_index('date'), bond.set_index('date'), gc.set_index('date'), forex.set_index('date') \
         ], axis=1, join='inner')
    returns = combined.loc[:, ['equity_ret', 'bond_ret', 'gc_ret', 'forex_ret']]
    returns.reset_index(inplace=True)
    returns['year'] = returns['date'].dt.year
    returns['month'] = returns['date'].dt.month
    returns['quarter'] = returns.month.apply(lambda v: 'Q1' if v <= 3 else 'Q2' if v <= 6 else 'Q3' if v <= 9 else 'Q4')
    returns['year_quarter'] = returns.year.astype(str) + '-' + returns.quarter.astype(str)
    returns.set_index('date', inplace=True)
    returns = returns.fillna('0')
    return (returns)
def gethistretdaily(returns):
    returns_act = returns.copy()
    returns_act = returns_act.drop(['year', 'month', 'quarter', 'year_quarter'], axis=1)
    returns_act = returns_act.fillna('0')
    daily_ret_his = returns_act.rolling(252).mean()
    daily_ret_his = daily_ret_his.fillna(0)
    return(daily_ret_his)

def readstage1data():
    rootd = Path(__file__).parent.parent.parent
    rootd = rootd.joinpath(rootd, "outputdata")
    rootd = rootd.joinpath(rootd, "Stage1")
    onlyfiles = sorted([f for f in listdir(rootd) if isfile(join(rootd, f))])
    filename = onlyfiles[0]
    data = pd.read_csv(str(rootd.joinpath(rootd, filename).absolute()), parse_dates=True)
    data['date'] = pd.to_datetime(data.date)
    return data
# covariance matrix from mgarch model
def get_covar_matrix_hat(x):
    X = np.array([x['equity_var_hat'], x['bond_equity_cov_hat'], x['equity_gold_cov_hat'], x['equity_forex_cov_hat'],
                  x['bond_equity_cov_hat'], x['bond_var_hat'], x['bond_gold_cov_hat'], x['bond_forex_cov_hat'],
                  x['equity_gold_cov_hat'], x['bond_gold_cov_hat'], x['gold_var_hat'], x['forex_gold_cov_hat'],
                  x['equity_forex_cov_hat'], x['bond_forex_cov_hat'], x['forex_gold_cov_hat'], x['forex_var_hat']])
    return opt.matrix(X, (4, 4))

def get_covar_matrix_his(x):
    X = np.array([x['equity_var_his'], x['bond_equity_his'], x['gold_equity_his'], x['forex_equity_his'],
                  x['bond_equity_his'], x['bond_var_his'], x['gold_bond_his'], x['forex_bond_his'],
                  x['gold_equity_his'], x['gold_bond_his'], x['gold_var_his'], x['gold_forex_his'],
                  x['forex_equity_his'], x['forex_bond_his'], x['gold_forex_his'], x['forex_var_his']])
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
    N = 1000
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
    # risks = [(sum(risks) / len(risks)) if math.isnan(x) else x for x in risks]
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
def find_weights(RISK, PORTFOLIOS, tgt_risk):
    '''
    RISK is a np.array() of risk levels
    RETURN is a np.array() of associated optimum returns
    Portfolios is weights
    '''
    assert len(RISK) == len(PORTFOLIOS)
    last = None
    for (risk, w) in zip(RISK, PORTFOLIOS):
        if risk > tgt_risk:
            return last
        last = w
    return last
def summary_metrics(R, rf_annual=0.01, show=False):
    pt = prettytable.PrettyTable(['metric', 'value'])
    avg = np.mean(R)
    std_dev = np.std(R)
    rf = rf_annual / 252
    count = len(R)
    sharpe = (avg - rf) / std_dev * np.sqrt(252)
    rpt = {'mean': avg,
           'std_dev': std_dev,
           'Sharpe_ratio': sharpe,
           'count': count}
    for (k, v) in rpt.items():
        pt.add_row([k, v])
    if show:
        print(pt)
    return rpt
def buildweightshat(inputstage1, df_daily_ret_act):
    mean_lookback_returns = []
    quarterly_cov_hat = []
    for x in inputstage1.to_dict('records'):
        sub = df_daily_ret_act[df_daily_ret_act.index < x['date']].tail(252)
        return_vec = sub[['equity_ret', 'bond_ret', 'gc_ret', 'forex_ret']].mean().values
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
        low_w = find_weights(table.risks, table.weights, 0.01)
        med_w = find_weights(table.risks, table.weights, 0.025)
        high_w = find_weights(table.risks, table.weights,0.04)
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
    weights_table['calculation_date'] = inputstage1['date']
    weights_table['year'] = weights_table['calculation_date'].dt.year
    weights_table['month'] = weights_table['calculation_date'].dt.month
    weights_table['quarter'] = weights_table.month.apply(
        lambda v: 'Q1' if v <= 3 else 'Q2' if v <= 6 else 'Q3' if v <= 9 else 'Q4')
    weights_table['year_quarter'] = weights_table.year.astype(str) + '-' + weights_table.quarter.astype(str)
    # weights_table = weights_table.drop(['year','month','quarter'], axis = 0)
    weights_table.set_index('year_quarter', inplace=True)
    weights_table = weights_table.fillna('0')
    return weights_table
def buildweightshis(inputstage1, df_daily_ret_act):
    mean_lookback_returns_his = []
    quarterly_cov_hat_his = []
    for x in inputstage1.to_dict('records'):
        sub = df_daily_ret_act[df_daily_ret_act.index < x['date']].tail(252)
        return_vec = sub[['equity_ret', 'bond_ret', 'gc_ret', 'forex_ret']].mean().values
        return_vec_cov = sub[['equity_ret', 'bond_ret', 'gc_ret', 'forex_ret']].cov()
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
        low_w = find_weights(table.risks, table.weights, 0.01)
        med_w = find_weights(table.risks, table.weights, 0.025)
        high_w = find_weights(table.risks, table.weights,0.04)
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
    weights_table_his['calculation_date'] = inputstage1['date']
    weights_table_his['year'] = weights_table_his['calculation_date'].dt.year
    weights_table_his['month'] = weights_table_his['calculation_date'].dt.month
    weights_table_his['quarter'] = weights_table_his.month.apply(
        lambda v: 'Q1' if v <= 3 else 'Q2' if v <= 6 else 'Q3' if v <= 9 else 'Q4')
    weights_table_his['year_quarter'] = weights_table_his.year.astype(str) + '-' + weights_table_his.quarter.astype(str)
    weights_table_his.set_index('year_quarter', inplace=True)
    weights_table_his = weights_table_his.fillna('0')
    return weights_table_his
def dailyportfolioreturn(returns_act, inputstage1_act, weights_table, weights_table_his):
    returns_act = returns_act[inputstage1_act.head(1).index[0]: inputstage1_act.tail(1).index[0]]
    weights_table_his = weights_table_his.drop(['year', 'month', 'quarter', 'calculation_date'], axis=1)
    weights_table = weights_table.drop(['year', 'month', 'quarter', 'calculation_date'], axis=1)
    returns_act['date'] = returns_act.index
    final_csv_temp = pd.merge(left=returns_act, right=weights_table_his, left_on='year_quarter',
                              right_on='year_quarter')
    final_csv_output = pd.merge(left=final_csv_temp, right=weights_table, left_on='year_quarter',
                                right_on='year_quarter')
    final_csv_output.set_index('date', inplace=True)
    return_arr = np.array(final_csv_output[['equity_ret', 'bond_ret', 'gc_ret', 'forex_ret']].T, dtype=float)
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
    return (final_csv_output)
def savecsvfile(csvstructure, filenamemask):
    if filenamemask.startswith('Stage2OutputReturns'):
        filename = r'outputdata\Stage2\Returns\SapiatStage2Out' + filenamemask + time.strftime("%Y%m%d%H%M%S") + '.csv'
    else:
        filename = r'outputdata\Stage2\Weights\SapiatStage2Out' +  filenamemask + time.strftime("%Y%m%d%H%M%S") + '.csv'
    csvstructure.to_csv(filename)
    print(f' Processing Complete, File :- {filename} has been created\r')
def main():
    bond, equity, forex, gc = loadassetfile()
    inputstage1 = readstage1data()
    inputstage1_act = inputstage1.copy()
    inputstage1_act.set_index('date', inplace=True)
    returns = getreturns(bond, equity, forex, gc)
    returns_act = returns.copy()
    returns_act = returns_act.drop(['year', 'month', 'quarter'], axis=1)
    df_daily_ret_act = returns[['equity_ret', 'bond_ret', 'gc_ret', 'forex_ret']]
    df_daily_ret_his = returns[['equity_ret', 'bond_ret', 'gc_ret', 'forex_ret']].rolling(252).mean()
    df_daily_ret_his['date'] = returns.index
    df_daily_ret_his = df_daily_ret_his.fillna('0')
    inputstage1['date'] = pd.to_datetime(inputstage1.date)
    weights_table_hat = buildweightshat(inputstage1, df_daily_ret_act)
    weights_table_his = buildweightshis(inputstage1, df_daily_ret_act)
    dailyportfolioreturncsv = dailyportfolioreturn(returns_act, inputstage1_act, weights_table_hat, weights_table_his)
    savecsvfile(dailyportfolioreturncsv, 'Stage2OutputReturns')
    savecsvfile(weights_table_hat, 'Stage2OutputWeightsHat')
    savecsvfile(weights_table_his, 'Stage2OutputWeightsHis')
    return dailyportfolioreturncsv
if __name__ == '__main__':
    dailyportfolioreturncsv = main()