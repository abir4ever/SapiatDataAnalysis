import pandas as pd
import numpy as np
import statsmodels.api as sm
import cvxopt as opt
from cvxopt import blas, solvers
import matplotlib.pylab as plt
from datetime import datetime
from datetime import timedelta

def read_csv_files():
    df = pd.read_csv('step1.csv')
    df_daily_ret_act = pd.read_csv('daily_returns_act.csv')
    df_daily_ret_his = pd.read_csv('daily_returns_his.csv')
    df_daily_ret_act['date'] = pd.to_datetime(df_daily_ret_act['date'])
    del df_daily_ret_act['Unnamed: 0']
    del df_daily_ret_his['Unnamed: 0']
    df.rename(columns={'Unnamed: 0':'date'},inplace=True)

    return df,df_daily_ret_act,df_daily_ret_his

# covariance matrix from linear regression model
def get_covar_matrix_hat(x):
    X = np.array([x['equity_var_hat'], x['bond_equity_cov_hat'],x['equity_gold_cov_hat'], x['equity_forex_cov_hat'],
                  x['bond_equity_cov_hat'],x['bond_var_hat'],x[ 'bond_gold_cov_hat'],x[ 'bond_forex_cov_hat'],
                  x['equity_gold_cov_hat'],x[ 'bond_gold_cov_hat'],x['gold_var_hat'],x['forex_gold_cov_hat'],
                 x['equity_forex_cov_hat'],x[ 'bond_forex_cov_hat'],x['forex_gold_cov_hat'],x['forex_var_hat']])
    return opt.matrix(X,(4,4))

# covariance matrix from historical values model
def get_covar_matrix_his(x):
    X = np.array([x['equity_var_his'], x['bond_equity_cov_his'],x['gold_equity_cov_his'], x['forex_equity_cov_his'],
                  x['bond_equity_cov_his'],x['bond_var_his'],x[ 'gold_bond_cov_his'],x[ 'forex_bond_cov_his'],
                  x['gold_equity_cov_his'],x[ 'gold_bond_cov_his'],x['gold_var_his'],x['gold_forex_cov_his'],
                 x['forex_equity_cov_his'],x[ 'forex_bond_cov_his'],x['gold_forex_cov_his'],x['forex_var_his']])
    return opt.matrix(X,(4,4))

# get optimal portfolio returns, risks and weights (100 combinations)

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
    portfolios = [solvers.qp(mu * S, -pbar, G, h, A, b)['x']
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(abs(m1[2] / m1[0]))
    # CALCULATE THE OPTIMAL PORTFOLIO
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

## Low, medium, high weights using LR cov

def get_weights_LR(df,df_daily_ret_act,lookback=252,low_tgt = 0.01,med_tgt = 0.025,high_tgt=0.04):
    # df contains quarterly returns sampled on last day of every quarter,
    #    covariance and variance - actual, historical and predicted
    mean_lookback_returns = []
    quarterly_cov_hat = []

    for x in df.to_dict('records'):
        sub = df_daily_ret_act[df_daily_ret_act['date']< x['date']].tail(lookback)
        return_vec = sub[[ 'equity_daily_ret','bond_daily_ret','gold_daily_ret','forex_daily_ret']].mean().values
        sigma = get_covar_matrix_hat(x)
        mean_lookback_returns += [return_vec]
        quarterly_cov_hat += [sigma]

    risks = []
    weights = []

    for ret, cov in zip(mean_lookback_returns, quarterly_cov_hat):
        w, re, ri, p = optimal_portfolio(ret, cov)
        risks += [ri]
        weights += [p]

    weights_table_hat = []

    for i in range(len(df)):
        table = {'risks': risks[i], 'weights': weights[i]}
        table = pd.DataFrame(table)
        table = table.sort_values(by=['risks'])
        low_w = find_weights(table.risks, table.weights, low_tgt)
        med_w = find_weights(table.risks, table.weights, med_tgt)
        high_w = find_weights(table.risks, table.weights, high_tgt)
        g = {'low_w': low_w,
             'med_w': med_w,
             'high_w': high_w}
        weights_table_hat += [g]

    weights_table_hat = pd.DataFrame(weights_table_hat)
    weights_table_hat['calculation_date'] = df['date']
    weights_table_hat['calculation_date'] = pd.to_datetime(weights_table_hat['calculation_date'])
    weights_table_hat['eff_start_date'] = weights_table_hat['calculation_date'] + timedelta(days=1)
    weights_table_hat['eff_end_date'] = weights_table_hat['eff_start_date'] + pd.offsets.QuarterEnd()
    weights_table_hat['eff_start_date'] = pd.to_datetime(weights_table_hat['eff_start_date'])
    weights_table_hat['eff_end_date'] = pd.to_datetime(weights_table_hat['eff_end_date'])

    return weights_table_hat

def get_weights_his(df,df_daily_ret_act,lookback=252,low_tgt = 0.01 ,med_tgt = 0.025,high_tgt=0.04):

    mean_lookback_returns_his = []
    quarterly_cov_his = []

    for x in df.to_dict('records'):
        sub = df_daily_ret_act[df_daily_ret_act['date'] < x['date']].tail(lookback)
        return_vec = sub[['equity_daily_ret', 'bond_daily_ret', 'gold_daily_ret', 'forex_daily_ret']].mean().values
        asset_cov = sub[['equity_daily_ret', 'bond_daily_ret', 'gold_daily_ret', 'forex_daily_ret']].cov()
        sigma = opt.matrix(asset_cov.values)
        mean_lookback_returns_his += [return_vec]
        quarterly_cov_his += [sigma]

    risks_his = []
    weights_his = []

    for ret, cov in zip(mean_lookback_returns_his, quarterly_cov_his):
        w, re, ri, p = optimal_portfolio(ret, cov)
        risks_his += [ri]
        weights_his += [p]

    weights_table_his = []
    for i in range(36):
        table = {'risks': risks_his[i], 'weights': weights_his[i]}
        table = pd.DataFrame(table)
        table = table.sort_values(by=['risks'])
        low_w = find_weights(table.risks, table.weights, low_tgt)
        med_w = find_weights(table.risks, table.weights, med_tgt)
        high_w = find_weights(table.risks, table.weights, high_tgt)
        g = {'low_w': low_w,
             'med_w': med_w,
             'high_w': high_w}
        weights_table_his += [g]

    weights_table_his = pd.DataFrame(weights_table_his)
    weights_table_his['calculation_date'] = df['date']
    weights_table_his['calculation_date'] = pd.to_datetime(weights_table_his['calculation_date'])
    weights_table_his['eff_start_date'] = weights_table_his['calculation_date'] + timedelta(days=1)
    weights_table_his['eff_end_date'] = weights_table_his['eff_start_date'] + pd.offsets.QuarterEnd()
    weights_table_his['eff_start_date'] = pd.to_datetime(weights_table_his['eff_start_date'])
    weights_table_his['eff_end_date'] = pd.to_datetime(weights_table_his['eff_end_date'])

    return weights_table_his

def get_hat_low_weights_portfolio_returns(weights_table_hat,df_daily_ret_act):

    weights_low_hat = weights_table_hat.copy()

    Equity_w = []
    Bond_w = []
    Gold_w = []
    Forex_w = []

    for w in weights_low_hat['low_w']:
        Equity_w += [float(w[0])]
        Bond_w += [float(w[1])]
        Gold_w += [float(w[2])]
        Forex_w += [float(w[3])]

    weights_low_hat['equity_low_w'] = Equity_w
    weights_low_hat['bond_low_w'] = Bond_w
    weights_low_hat['gold_low_w'] = Gold_w
    weights_low_hat['forex_low_w'] = Forex_w

    header = ['date',
              'equity_daily_ret', 'bond_daily_ret', 'gold_daily_ret', 'forex_daily_ret',
              'equity_hat_low_w', 'bond_hat_low_w', 'gold_hat_low_w', 'forex_hat_low_w',
              'ret']

    report_1 = pd.DataFrame(columns=header)

    for x in weights_low_hat.to_dict('records'):

        W = [x['equity_low_w'], x['bond_low_w'], x['gold_low_w'], x['forex_low_w']]
        start = x['eff_start_date']
        end = x['eff_end_date']

        df_sub_daily = df_daily_ret_act[(df_daily_ret_act['date'] >= start) & (df_daily_ret_act['date'] <= end)]

        sub_head = ['equity_hat_low_w', 'bond_hat_low_w', 'gold_hat_low_w', 'forex_hat_low_w']
        W_prime = [W] * len(df_sub_daily)

        df_weights_tmp = pd.DataFrame(columns=sub_head, data=W_prime)
        df_weights_tmp['date'] = df_sub_daily['date'].values

        df_qtr = pd.concat([df_sub_daily.set_index('date'), df_weights_tmp.set_index('date')], axis=1)
        df_qtr['weighted_ret_equity'] = df_qtr['equity_hat_low_w'] * df_qtr['equity_daily_ret']
        df_qtr['weighted_ret_bond'] = df_qtr['bond_hat_low_w'] * df_qtr['bond_daily_ret']
        df_qtr['weighted_ret_gold'] = df_qtr['gold_hat_low_w'] * df_qtr['gold_daily_ret']
        df_qtr['weighted_ret_forex'] = df_qtr['forex_hat_low_w'] * df_qtr['forex_daily_ret']
        df_qtr['ret'] = df_qtr['weighted_ret_equity'] + df_qtr['weighted_ret_bond'] + \
                        df_qtr['weighted_ret_gold'] + df_qtr['weighted_ret_forex']
        df_qtr = df_qtr[['equity_daily_ret', 'bond_daily_ret', 'gold_daily_ret', 'forex_daily_ret',
                         'equity_hat_low_w', 'bond_hat_low_w', 'gold_hat_low_w', 'forex_hat_low_w',
                         'ret']]
        df_qtr.reset_index(inplace=True)
        report_1 = report_1.append(df_qtr)

    report_1.to_csv('Hat_low_port_returns.csv')
    return report_1

def get_hat_med_weights_portfolio_returns(weights_table_hat,df_daily_ret_act):

    weights_med_hat = weights_table_hat.copy()

    Equity_w = []
    Bond_w = []
    Gold_w = []
    Forex_w = []

    for w in weights_med_hat['med_w']:
        Equity_w += [float(w[0])]
        Bond_w += [float(w[1])]
        Gold_w += [float(w[2])]
        Forex_w += [float(w[3])]

    weights_med_hat['equity_med_w'] = Equity_w
    weights_med_hat['bond_med_w'] = Bond_w
    weights_med_hat['gold_med_w'] = Gold_w
    weights_med_hat['forex_med_w'] = Forex_w

    header = ['date',
              'equity_daily_ret', 'bond_daily_ret', 'gold_daily_ret', 'forex_daily_ret',
              'equity_hat_med_w', 'bond_hat_med_w', 'gold_hat_med_w', 'forex_hat_med_w',
              'ret']

    report_2 = pd.DataFrame(columns=header)

    for x in weights_med_hat.to_dict('records'):

        W = [x['equity_med_w'], x['bond_med_w'], x['gold_med_w'], x['forex_med_w']]
        start = x['eff_start_date']
        end = x['eff_end_date']

        df_sub_daily = df_daily_ret_act[
            (df_daily_ret_act['date'] >= start) & (df_daily_ret_act['date'] <= end)]

        sub_head = ['equity_hat_med_w', 'bond_hat_med_w', 'gold_hat_med_w', 'forex_hat_med_w']
        W_prime = [W] * len(df_sub_daily)
        df_weights_tmp = pd.DataFrame(columns=sub_head, data=W_prime)
        df_weights_tmp['date'] = df_sub_daily['date'].values
        df_qtr = pd.concat([df_sub_daily.set_index('date'), df_weights_tmp.set_index('date')], axis=1)
        df_qtr['weighted_ret_equity'] = df_qtr['equity_hat_med_w'] * df_qtr['equity_daily_ret']
        df_qtr['weighted_ret_bond'] = df_qtr['bond_hat_med_w'] * df_qtr['bond_daily_ret']
        df_qtr['weighted_ret_gold'] = df_qtr['gold_hat_med_w'] * df_qtr['gold_daily_ret']
        df_qtr['weighted_ret_forex'] = df_qtr['forex_hat_med_w'] * df_qtr['forex_daily_ret']
        df_qtr['ret'] = df_qtr['weighted_ret_equity'] + df_qtr['weighted_ret_bond'] + \
                        df_qtr['weighted_ret_gold'] + df_qtr['weighted_ret_forex']
        df_qtr = df_qtr[['equity_daily_ret', 'bond_daily_ret', 'gold_daily_ret', 'forex_daily_ret',
                         'equity_hat_med_w', 'bond_hat_med_w', 'gold_hat_med_w', 'forex_hat_med_w',
                         'ret']]
        df_qtr.reset_index(inplace=True)
        report_2 = report_2.append(df_qtr)

    report_2.to_csv('Hat_med_port_returns.csv')
    return report_2

def get_hat_high_weights_portfolio_returns(weights_table_hat,df_daily_ret_act):

    weights_high_hat = weights_table_hat.copy()

    Equity_w = []
    Bond_w = []
    Gold_w = []
    Forex_w = []

    for w in weights_high_hat['high_w']:

        Equity_w += [float(w[0])]
        Bond_w += [float(w[1])]
        Gold_w += [float(w[2])]
        Forex_w += [float(w[3])]

    weights_high_hat['equity_high_w'] = Equity_w
    weights_high_hat['bond_high_w'] = Bond_w
    weights_high_hat['gold_high_w'] = Gold_w
    weights_high_hat['forex_high_w'] = Forex_w

    header = ['date',
              'equity_daily_ret', 'bond_daily_ret', 'gold_daily_ret', 'forex_daily_ret',
              'equity_hat_high_w', 'bond_hat_high_w', 'gold_hat_high_w', 'forex_hat_high_w',
              'ret']

    report_3 = pd.DataFrame(columns=header)

    for x in weights_high_hat.to_dict('records'):
        W = [x['equity_high_w'], x['bond_high_w'], x['gold_high_w'], x['forex_high_w']]
        start = x['eff_start_date']
        end = x['eff_end_date']

        df_sub_daily = df_daily_ret_act[(df_daily_ret_act['date'] >= start) & \
                                     (df_daily_ret_act['date'] <= end)]

        sub_head = ['equity_hat_high_w', 'bond_hat_high_w', 'gold_hat_high_w', 'forex_hat_high_w']
        W_prime = [W] * len(df_sub_daily)
        df_weights_tmp = pd.DataFrame(columns=sub_head, data=W_prime)
        df_weights_tmp['date'] = df_sub_daily['date'].values
        df_qtr = pd.concat([df_sub_daily.set_index('date'), df_weights_tmp.set_index('date')], axis=1)
        df_qtr['weighted_ret_equity'] = df_qtr['equity_hat_high_w'] * df_qtr['equity_daily_ret']
        df_qtr['weighted_ret_bond'] = df_qtr['bond_hat_high_w'] * df_qtr['bond_daily_ret']
        df_qtr['weighted_ret_gold'] = df_qtr['gold_hat_high_w'] * df_qtr['gold_daily_ret']
        df_qtr['weighted_ret_forex'] = df_qtr['forex_hat_high_w'] * df_qtr['forex_daily_ret']
        df_qtr['ret'] = df_qtr['weighted_ret_equity'] + df_qtr['weighted_ret_bond'] + \
                        df_qtr['weighted_ret_gold'] + df_qtr['weighted_ret_forex']
        df_qtr = df_qtr[['equity_daily_ret', 'bond_daily_ret', 'gold_daily_ret', 'forex_daily_ret',
                         'equity_hat_high_w', 'bond_hat_high_w', 'gold_hat_high_w', 'forex_hat_high_w',
                         'ret']]
        df_qtr.reset_index(inplace=True)
        report_3 = report_3.append(df_qtr)

    report_3.to_csv('Hat_high_port_returns.csv')
    return report_3

def get_his_low_weights_portfolio_returns(weights_table_his,df_daily_ret_act):

    weights_low_his = weights_table_his.copy()

    Equity_w = []
    Bond_w = []
    Gold_w = []
    Forex_w = []

    for w in weights_low_his['low_w']:
        Equity_w += [float(w[0])]
        Bond_w += [float(w[1])]
        Gold_w += [float(w[2])]
        Forex_w += [float(w[3])]

    weights_low_his['equity_low_w'] = Equity_w
    weights_low_his['bond_low_w'] = Bond_w
    weights_low_his['gold_low_w'] = Gold_w
    weights_low_his['forex_low_w'] = Forex_w

    header = ['date',
              'equity_daily_ret', 'bond_daily_ret', 'gold_daily_ret', 'forex_daily_ret',
                  'equity_his_low_w', 'bond_his_low_w', 'gold_his_low_w', 'forex_his_low_w',
                  'ret']
    report_4 = pd.DataFrame(columns=header)

    for x in weights_low_his.to_dict('records'):
        W = [x['equity_low_w'], x['bond_low_w'], x['gold_low_w'], x['forex_low_w']]
        start = x['eff_start_date']
        end = x['eff_end_date']

        df_sub_daily = df_daily_ret_act[(df_daily_ret_act['date'] >= start) & (df_daily_ret_act['date'] <= end)]

        sub_head = ['equity_his_low_w', 'bond_his_low_w', 'gold_his_low_w', 'forex_his_low_w']
        W_prime = [W] * len(df_sub_daily)
        df_weights_tmp = pd.DataFrame(columns=sub_head, data=W_prime)
        df_weights_tmp['date'] = df_sub_daily['date'].values
        df_qtr = pd.concat([df_sub_daily.set_index('date'), df_weights_tmp.set_index('date')], axis=1)
        df_qtr['weighted_ret_equity'] = df_qtr['equity_his_low_w'] * df_qtr['equity_daily_ret']
        df_qtr['weighted_ret_bond'] = df_qtr['bond_his_low_w'] * df_qtr['bond_daily_ret']
        df_qtr['weighted_ret_gold'] = df_qtr['gold_his_low_w'] * df_qtr['gold_daily_ret']
        df_qtr['weighted_ret_forex'] = df_qtr['forex_his_low_w'] * df_qtr['forex_daily_ret']
        df_qtr['ret'] = df_qtr['weighted_ret_equity'] + df_qtr['weighted_ret_bond'] + \
                        df_qtr['weighted_ret_gold'] + df_qtr['weighted_ret_forex']
        df_qtr = df_qtr[['equity_daily_ret', 'bond_daily_ret', 'gold_daily_ret', 'forex_daily_ret',
                         'equity_his_low_w', 'bond_his_low_w', 'gold_his_low_w', 'forex_his_low_w',
                         'ret']]
        df_qtr.reset_index(inplace=True)
        report_4 = report_4.append(df_qtr)

    report_4.to_csv('His_low_port_returns.csv')
    return report_4

def get_his_med_weights_portfolio_returns(weights_table_his,df_daily_ret_act):

    weights_med_his = weights_table_his.copy()

    Equity_w = []
    Bond_w = []
    Gold_w = []
    Forex_w = []

    for w in weights_med_his['med_w']:
        Equity_w += [float(w[0])]
        Bond_w += [float(w[1])]
        Gold_w += [float(w[2])]
        Forex_w += [float(w[3])]

    weights_med_his['equity_med_w'] = Equity_w
    weights_med_his['bond_med_w'] = Bond_w
    weights_med_his['gold_med_w'] = Gold_w
    weights_med_his['forex_med_w'] = Forex_w

    header = ['date',
              'equity_daily_ret', 'bond_daily_ret', 'gold_daily_ret', 'forex_daily_ret',
              'equity_his_med_w', 'bond_his_med_w', 'gold_his_med_w', 'forex_his_med_w',
              'ret']

    report_5 = pd.DataFrame(columns=header)

    for x in weights_med_his.to_dict('records'):
        W = [x['equity_med_w'], x['bond_med_w'], x['gold_med_w'], x['forex_med_w']]
        start = x['eff_start_date']
        end = x['eff_end_date']

        df_sub_daily = df_daily_ret_act[
            (df_daily_ret_act['date'] >= start) & (df_daily_ret_act['date'] <= end)]

        sub_head = ['equity_his_med_w', 'bond_his_med_w', 'gold_his_med_w', 'forex_his_med_w']
        W_prime = [W] * len(df_sub_daily)
        df_weights_tmp = pd.DataFrame(columns=sub_head, data=W_prime)
        df_weights_tmp['date'] = df_sub_daily['date'].values
        df_qtr = pd.concat([df_sub_daily.set_index('date'), df_weights_tmp.set_index('date')], axis=1)
        df_qtr['weighted_ret_equity'] = df_qtr['equity_his_med_w'] * df_qtr['equity_daily_ret']
        df_qtr['weighted_ret_bond'] = df_qtr['bond_his_med_w'] * df_qtr['bond_daily_ret']
        df_qtr['weighted_ret_gold'] = df_qtr['gold_his_med_w'] * df_qtr['gold_daily_ret']
        df_qtr['weighted_ret_forex'] = df_qtr['forex_his_med_w'] * df_qtr['forex_daily_ret']
        df_qtr['ret'] = df_qtr['weighted_ret_equity'] + df_qtr['weighted_ret_bond'] + \
                        df_qtr['weighted_ret_gold'] + df_qtr['weighted_ret_forex']
        df_qtr = df_qtr[['equity_daily_ret', 'bond_daily_ret', 'gold_daily_ret', 'forex_daily_ret',
                         'equity_his_med_w', 'bond_his_med_w', 'gold_his_med_w', 'forex_his_med_w',
                         'ret']]
        df_qtr.reset_index(inplace=True)
        report_5 = report_5.append(df_qtr)

    report_5.to_csv('His_med_port_returns.csv')
    return report_5

def get_his_high_weights_portfolio_returns(weights_table_his,df_daily_ret_act):

    weights_high_his = weights_table_his.copy()

    Equity_w = []
    Bond_w = []
    Gold_w = []
    Forex_w = []

    for w in weights_high_his['high_w']:
        Equity_w += [float(w[0])]
        Bond_w += [float(w[1])]
        Gold_w += [float(w[2])]
        Forex_w += [float(w[3])]

    weights_high_his['equity_high_w'] = Equity_w
    weights_high_his['bond_high_w'] = Bond_w
    weights_high_his['gold_high_w'] = Gold_w
    weights_high_his['forex_high_w'] = Forex_w

    header = ['date',
              'equity_daily_ret', 'bond_daily_ret', 'gold_daily_ret', 'forex_daily_ret',
              'equity_his_high_w', 'bond_his_high_w', 'gold_his_high_w', 'forex_his_high_w',
              'ret']

    report_6 = pd.DataFrame(columns=header)
    for x in weights_high_his.to_dict('records'):
        W = [x['equity_high_w'], x['bond_high_w'], x['gold_high_w'], x['forex_high_w']]
        start = x['eff_start_date']
        end = x['eff_end_date']

        df_sub_daily = df_daily_ret_act[
            (df_daily_ret_act['date'] >= start) & (df_daily_ret_act['date'] <= end)]

        sub_head = ['equity_his_high_w', 'bond_his_high_w', 'gold_his_high_w', 'forex_his_high_w']
        W_prime = [W] * len(df_sub_daily)
        df_weights_tmp = pd.DataFrame(columns=sub_head, data=W_prime)
        df_weights_tmp['date'] = df_sub_daily['date'].values
        df_qtr = pd.concat([df_sub_daily.set_index('date'), df_weights_tmp.set_index('date')], axis=1)
        df_qtr['weighted_ret_equity'] = df_qtr['equity_his_high_w'] * df_qtr['equity_daily_ret']
        df_qtr['weighted_ret_bond'] = df_qtr['bond_his_high_w'] * df_qtr['bond_daily_ret']
        df_qtr['weighted_ret_gold'] = df_qtr['gold_his_high_w'] * df_qtr['gold_daily_ret']
        df_qtr['weighted_ret_forex'] = df_qtr['forex_his_high_w'] * df_qtr['forex_daily_ret']
        df_qtr['ret'] = df_qtr['weighted_ret_equity'] + df_qtr['weighted_ret_bond'] + \
                        df_qtr['weighted_ret_gold'] + df_qtr['weighted_ret_forex']
        df_qtr = df_qtr[['equity_daily_ret', 'bond_daily_ret', 'gold_daily_ret', 'forex_daily_ret',
                         'equity_his_high_w', 'bond_his_high_w', 'gold_his_high_w', 'forex_his_high_w',
                         'ret']]
        df_qtr.reset_index(inplace=True)
        report_6 = report_6.append(df_qtr)

    report_6.to_csv('His_high_port_returns.csv')
    return report_6
