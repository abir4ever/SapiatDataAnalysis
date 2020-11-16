import pandas as pd
from pathlib import Path
def loadfile():
    rootd = Path(__file__).parent.parent.parent.parent
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
if __name__ == '__main__':
    loadfile()