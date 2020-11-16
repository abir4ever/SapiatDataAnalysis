# importing necessary libraries
import pandas as pd
import numpy as np
import time
from rpy2.robjects import pandas2ri
import rpy2
from rpy2.robjects import numpy2ri
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
def loadFile():
    bond = pd.read_csv(r'Data\bond.csv', parse_dates=True)
    # bond returns are calculated in the excel file itself (summation of price change and interest)
    bond['date'] = pd.to_datetime(bond.date)
    equity = pd.read_csv(r'Data\equity.csv', parse_dates=True)
    equity['equity_ret'] = equity.adj_close.pct_change()
    equity['date'] = pd.to_datetime(equity.date)
    forex = pd.read_csv(r'Data\forex.csv', parse_dates=True)
    forex['forex_ret'] = forex.adj_close.pct_change()
    forex['date'] = pd.to_datetime(forex.date)
    gc = pd.read_csv(r'Data\gc.csv', parse_dates=True)
    gc['gc_ret'] = gc.adj_close.pct_change()
    gc['date'] = pd.to_datetime(gc.date)
    return (bond, equity, forex, gc)
def getReturns(bond, equity, forex, gc):
    combined = pd.concat([bond.set_index('date'), equity.set_index('date'), forex.set_index('date'), \
                          gc.set_index('date')], axis=1, join='inner')
    returns = combined.loc[:, ['bond_ret', 'equity_ret', 'forex_ret', 'gc_ret']]
    returns.reset_index(inplace=True)
    returns['year'] = returns['date'].dt.year
    returns['month'] = returns['date'].dt.month
    returns['quarter'] = returns.month.apply(lambda v: 'Q1' if v <= 3 else 'Q2' if v <= 6 else 'Q3' if v <= 9 else 'Q4')
    returns['year_quarter'] = returns.year.astype(str) + '-' + returns.quarter.astype(str)
    returns.set_index('date', inplace=True)
    return (returns)
def getQrtReturnsVarHat(equity,bond,forex,gc):
    start_date = '1982-03-31'
    end_date = '2020-03-31'
    bond = bond.set_index('date').asfreq('d').fillna(method='ffill')
    bond['price'] = 1 - (bond['adj_close'] / 100 * 90 / 360)
    bond_qrt_return = bond['price'].resample('Q').ffill().pct_change()
    bond_qrt_return = pd.DataFrame(bond_qrt_return)
    bond_qrt_return.columns = ['price_return']
    bond_df = pd.concat([bond, bond_qrt_return], axis=1, join='inner')
    bond_df['int_return'] = bond_df['adj_close'].shift(1) * 90 / 360 / 100
    bond_df['total_qt_return'] = bond_df['price_return'] + bond_df['int_return']
    eq_qrt_ret = equity.set_index('date')['adj_close'].resample('Q').ffill().pct_change()
    bond_qrt_ret = bond_df['total_qt_return'][start_date:end_date]
    gold_qrt_ret = gc.set_index('date')['adj_close'].resample('Q').ffill().pct_change()
    forex_qrt_ret = forex.set_index('date')['adj_close'].resample('Q').ffill().pct_change()
    qrt_ret = pd.DataFrame(data=(eq_qrt_ret, bond_qrt_ret, gold_qrt_ret, forex_qrt_ret))
    qrt_ret = qrt_ret.transpose()
    qrt_ret.columns = ['equity_qrt_ret_act', 'bond_qrt_ret_act', 'gold_qrt_ret_act', 'forex_qrt_ret_act']
    qrt_ret['equity_var_act'] = (qrt_ret['equity_qrt_ret_act'] - qrt_ret['equity_qrt_ret_act'].mean()) ** 2
    qrt_ret['bond_var_act'] = (qrt_ret['bond_qrt_ret_act'] - qrt_ret['bond_qrt_ret_act'].mean()) ** 2
    qrt_ret['gold_var_act'] = (qrt_ret['bond_qrt_ret_act'] - qrt_ret['bond_qrt_ret_act'].mean()) ** 2
    qrt_ret['forex_var_act'] = (qrt_ret['bond_qrt_ret_act'] - qrt_ret['bond_qrt_ret_act'].mean()) ** 2
    qrt_ret['equity_hat_ret'] = qrt_ret['equity_qrt_ret_act'].rolling(4).mean()
    qrt_ret['bond_hat_ret'] = qrt_ret['bond_qrt_ret_act'].rolling(4).mean()
    qrt_ret['forex_hat_ret'] = qrt_ret['forex_qrt_ret_act'].rolling(4).mean()
    qrt_ret['gold_hat_ret'] = qrt_ret['gold_qrt_ret_act'].rolling(4).mean()
    qrt_ret['equity_var_his'] = qrt_ret['equity_var_act'].rolling(4).mean()
    qrt_ret['bond_var_his'] = qrt_ret['bond_var_act'].rolling(4).mean()
    qrt_ret['forex_var_his'] = qrt_ret['forex_var_act'].rolling(4).mean()
    qrt_ret['gold_var_his'] = qrt_ret['gold_var_act'].rolling(4).mean()
    qrt_ret = qrt_ret[start_date:end_date]
    return(qrt_ret.fillna('0'))
def getRetQuarterCov(returns):
    # creating a dictionary with keys as the quarter and values as correlation between assets for that quarter
    CORR_YQ = {}
    COV_YQ = {}
    for year in range(1982, 2021):
        for quarter in range(1, 5):
            sub3 = returns[(returns['year'] == year) & (returns['quarter'] == ('Q' + str(quarter)))][['equity_ret', 'bond_ret', 'forex_ret', 'gc_ret']]
            sub3 = sub3.fillna('0')
            sub3Cov = sub3.cov()
            COV_YQ[str(year) + 'Q' + str(quarter)] = sub3Cov
    # Creating a list with quarter names and relevant correlation values
    Xq = []
    Yq = []
    #print(type(COV_YQ))
    for year_quarter in sorted(COV_YQ.keys()):
        cov_ravelled = np.ravel(COV_YQ[year_quarter])
        if (len(cov_ravelled)) >= 16:
            Xq += [year_quarter]
            Yq += [cov_ravelled[[(1, 2, 3, 6, 7, 11)]]]
        # Creating a dataframe with quarter lables and relevant correlation values
    pairs_q = {1: ('bond', 'equity'), 2: ('forex', 'equity'), 3: ('gold', 'equity'),
               6: ('forex', 'bond'), 7: ('gold', 'bond'), 11: ('gold', 'forex')}
    report_q = []
    for year_quarter in sorted(COV_YQ.keys()):
        cov_q = np.ravel(COV_YQ[year_quarter])
        if (len(cov_q)) >= 16:
            for i in (1, 2, 3, 6, 7, 11):
                r = {'year_quarter': year_quarter,
                     'symbol_1': pairs_q[i][0],
                     'symbol_2': pairs_q[i][1],
                     'cov': cov_q[i]}
                report_q += [r]
    df_cov_quarter = pd.DataFrame(report_q)
    # Dataframe with combination value
    df_cov_quarter['combination'] = df_cov_quarter.symbol_1 + "_" + df_cov_quarter.symbol_2 + "_act"
    data_q = pd.DataFrame(columns=df_cov_quarter.combination.unique(), index=df_cov_quarter.year_quarter.unique(),
                          data=Yq)
    data_qt = data_q.transpose()
    data_qt = data_qt.iloc[:, :-2]
    data_q = data_q.iloc[:-2, :]
    data_q['bond_equity_his']  = data_q['bond_equity_act'].rolling(4).mean()
    data_q['forex_equity_his'] = data_q['forex_equity_act'].rolling(4).mean()
    data_q['gold_equity_his']  = data_q['gold_equity_act'].rolling(4).mean()
    data_q['forex_bond_his']   = data_q['forex_bond_act'].rolling(4).mean()
    data_q['gold_bond_his']    = data_q['gold_bond_act'].rolling(4).mean()
    data_q['gold_forex_his']   = data_q['gold_forex_act'].rolling(4).mean()
    data_q = data_q.rename_axis("year_quarter")
    return(data_q.fillna('0'))

def dccmgarch(pd_rets, n_look_ahead=1):
    base = importr('base')
    rmgarch = importr('rmgarch')
    rugarch = importr('rugarch')
    pandas2ri.activate() # Start of R
    rpy2.robjects.numpy2ri.activate()
    r_rets = pandas2ri.py2ri(pd_rets) # convert the daily returns from pandas dataframe in Python to dataframe in R
    n = base.dim(r_rets)[1]
    ugarch_spec = rugarch.ugarchspec(mean_model = robjects.r('list(armaOrder = c(1,0))'))
    mgarch_spec = base.replicate( n, ugarch_spec)
    uspec_4 = rugarch.multispec(mgarch_spec)
    multf = rugarch.multifit(uspec_4, r_rets)
    spec = rmgarch.dccspec(uspec = uspec_4, dccOrder = robjects.r('c(1, 1)'), distribution = 'mvnorm')
    fit = rmgarch.dccfit(spec, data = r_rets, fit_control = robjects.r('list(eval.se = TRUE)'), fit = multf)
    forecasts = rmgarch.dccforecast(fit, n_ahead = n_look_ahead)
    r_forecast_cov = rmgarch.rcov(forecasts)
    numpy2ri.deactivate() # End of R
    #rpy2.robjects.numpy2ri.deactivate()
    # access and transform the covariance matrices in R format
    n_cols = pd_rets.shape[1] # get the number of assets in pd_rets
    n_elements = n_cols*n_cols # the number of elements in each covariance matrix
    n_matrix = int(len(r_forecast_cov[0])/(n_elements))
    # sum the daily forecasted covariance matrices
    cov_matrix = 0
    for i in range(n_matrix):
        i_matrix = np.array([v for v in r_forecast_cov[0][i*n_elements:(i+1)*n_elements]])
        i_matrix = i_matrix.reshape(n_cols,n_cols)
        cov_matrix += i_matrix
    return(cov_matrix)
def getTrainTestDates():
    train_start_date = '1982-09-30'
    train_end_date = '2007-12-31'
    test_start_date = '2008-03-31'
    test_end_date = '2016-12-31'
    return (train_start_date, train_end_date,test_start_date,test_end_date)
def getQtrCovVarMgarch(returns, train_start_date, train_end_date,test_start_date,test_end_date):
    mgarchreturn = returns.copy()
    mgarchreturn = mgarchreturn.drop(['year','month', 'quarter', 'year_quarter'], axis=1)
    mgarchreturn = mgarchreturn.resample('Q').ffill()
    train_set = mgarchreturn.loc[train_start_date:train_end_date]
    test_set = mgarchreturn.loc[test_start_date:test_end_date]
    Cov_matrix = []
    prediction_line = pd.DataFrame(
        columns=['date', 'bond_var_hat', 'bond_equity_cov_hat', 'equity_var_hat', 'bond_forex_cov_hat',
                 'equity_forex_cov_hat', 'forex_var_hat', 'bond_gold_cov_hat', 'equity_gold_cov_hat',
                 'forex_gold_cov_hat', 'gold_var_hat'])
    prediction = pd.DataFrame(
        columns=['date', 'bond_var_hat', 'bond_equity_cov_hat', 'equity_var_hat', 'bond_forex_cov_hat',
                 'equity_forex_cov_hat', 'forex_var_hat', 'bond_gold_cov_hat', 'equity_gold_cov_hat',
                 'forex_gold_cov_hat', 'gold_var_hat'])
    for i, row in enumerate(test_set.values):
        cov_matrix = dccmgarch(train_set, 1)
        Cov_matrix.append(cov_matrix.tolist())
        list_of_dates = []
        list_of_dates.append(test_set.index[i])
        prediction_line['date'] = list_of_dates
        prediction_line[['bond_var_hat', 'bond_equity_cov_hat', 'equity_var_hat', 'bond_forex_cov_hat',
                         'equity_forex_cov_hat', 'forex_var_hat', 'bond_gold_cov_hat', 'equity_gold_cov_hat',
                        'forex_gold_cov_hat', 'gold_var_hat']] = cov_matrix[
            np.tril_indices(len(train_set.columns))].tolist()
        prediction = prediction.append(prediction_line)
        one_qtr = test_set[test_set.index == test_set.index[i]]
        train_set = train_set.append(one_qtr)
        print(f' Predicting MGARCH Model for Quarter {test_set.index[i]} {i+1} of {len(test_set.index)}\r', end="")
    prediction.reset_index(inplace=True)
    prediction['year'] = prediction['date'].dt.year
    prediction['month'] = prediction['date'].dt.month
    prediction['quarter'] = prediction.month.apply(lambda v: 'Q1' if v <= 3 else 'Q2' if v <= 6 else 'Q3' if v <= 9 else 'Q4')
    prediction['year_quarter'] = prediction.year.astype(str) +  prediction.quarter.astype(str)
    prediction['date'] = pd.to_datetime(prediction.date)
    prediction.set_index('date', inplace=True)
    prediction = prediction.drop('index', axis=1)
    prediction.index.freq = 'Q'
    return prediction
def buildfinaldf(prediction, qrtretvarhat,retquartercov):
    csvstructure = pd.DataFrame(
                            columns =['date','equity_qrt_ret_act', 'bond_qrt_ret_act', 'forex_qrt_ret_act',
                                      'gold_qrt_ret_act','bond_equity_act', 'forex_equity_act', 'gold_equity_act',
                                      'forex_bond_act', 'gold_bond_act', 'gold_forex_act', 'equity_var_act',
                                      'bond_var_act', 'gold_var_act', 'forex_var_act', 'equity_hat_ret', 'bond_hat_ret',
                                      'gold_hat_ret', 'forex_hat_ret','bond_equity_his', 'forex_equity_his',
                                      'gold_equity_his', 'forex_bond_his', 'gold_bond_his', 'gold_forex_his',
                                      'bond_equity_cov_hat', 'bond_forex_cov_hat', 'equity_forex_cov_hat',
                                      'bond_gold_cov_hat', 'equity_gold_cov_hat','forex_gold_cov_hat','equity_var_hat',
                                      'bond_var_hat', 'gold_var_hat', 'forex_var_hat'])
    prediction.reset_index(inplace=True)
    prediction['date'] = pd.to_datetime(prediction.date)
    csvstructure = pd.merge(left=prediction, right=retquartercov, left_on='year_quarter', right_on='year_quarter')
    csvstructure.reset_index(inplace=True)
    csvstructure['date'] = pd.to_datetime(csvstructure.date)
    csvstructure.set_index('date', inplace=True)
    csvstructure = csvstructure.drop('index', axis=1)
    csvstructure = pd.concat([csvstructure, qrtretvarhat], axis=1, join='inner')
    csvstructure = csvstructure.drop(['year','month','quarter','year_quarter'], axis=1)
    csvstructurefinal = csvstructure[[# Returns - actual - Done qrtretvarhat
     'equity_qrt_ret_act', 'bond_qrt_ret_act', 'forex_qrt_ret_act', 'gold_qrt_ret_act',
     # Covariance - actual -> Done
     'bond_equity_act', 'forex_equity_act', 'gold_equity_act', 'forex_bond_act', 'gold_bond_act', 'gold_forex_act',
     # Variance - actual -> Done
     'equity_var_act', 'bond_var_act', 'gold_var_act', 'forex_var_act',
     # Returns - rolling -> Done
     'equity_hat_ret', 'bond_hat_ret', 'gold_hat_ret', 'forex_hat_ret',
     # Covariance - rolling->Done
     'bond_equity_his', 'forex_equity_his', 'gold_equity_his', 'forex_bond_his', 'gold_bond_his', 'gold_forex_his',
     # Covariance Mgarch
     'bond_equity_cov_hat', 'bond_forex_cov_hat', 'equity_forex_cov_hat', 'bond_gold_cov_hat', 'equity_gold_cov_hat',
     'forex_gold_cov_hat','equity_var_hat', 'bond_var_hat', 'gold_var_hat', 'forex_var_hat']]
    return(csvstructurefinal)
def savecsvfile(csvstructure):
    filename = r'Output\Stage1\Stage1Out' + time.strftime("%Y%m%d%H%M%S") +'.csv'
    csvstructure.to_csv(filename)
def DataPrepMain():
    bond, equity, forex, gc = loadFile()
    returns = getReturns(bond, equity, forex, gc)
    retquartercov = getRetQuarterCov(returns)
    qrtretvarhat = getQrtReturnsVarHat(equity, bond, forex, gc)
    train_start_date, train_end_date,test_start_date,test_end_date = getTrainTestDates()
    prediction = getQtrCovVarMgarch(returns,train_start_date, train_end_date,test_start_date,test_end_date)
    csvstructure = buildfinaldf(prediction, qrtretvarhat,retquartercov)
    savecsvfile(csvstructure)
    return(bond, equity, forex, gc,returns,retquartercov,qrtretvarhat,
           train_start_date, train_end_date,test_start_date,test_end_date,
           prediction,csvstructure)
if __name__ == '__main__':
    DataPrepMain()

