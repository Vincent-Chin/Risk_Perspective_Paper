import pandas as pd
import os

import pull_history as ph

###########
# HELPER FUNCTIONS
###########
def populate_data(symbols, data_directory, initial_date, final_date):

    if not os.path.exists(data_directory):
        os.mkdir(data_directory)

    response_data = {}
    for symbol in symbols:
        filename = data_directory + symbol + '.csv'
        if os.path.exists(filename):
            this_data = pd.read_csv(filename, index_col=0)
        else:
            this_data = ph.getEquity(symbol, initial_date, final_date)
            this_data.to_csv(filename)
        response_data[symbol] = this_data

    return response_data


###########
# MAIN LOGIC
###########
def main():
    # PROCESS
    # -------
    # 1. stage data - pull from csvs
    # 2. compute study values (moving averages - 20 period, 50 period, 200 period)
    # 3. enumerate search universe
    #
    # 4. for each search metric:
    #   a. for each date of testing period:
    #       1. isolate data bounds using lookback
    #       3. apply filtering constraints for each permutation of the search universe
    #       4. cache signal performance on a returns basis
    #       5. choose best performer on day n-1 for active signals on day n, by search metric
    #       6. aggregate signals onto a outsample and insample signals file for the search metric
    #
    # 5. for each signals file:
    #   a. generate a performance comparison with some descriptive statistics for each
    #   b. plot the cumulative PLs on a graph
    #

    # --- 1. INITIALIZATION ---
    # https://en.wikipedia.org/wiki/S%26P_100 - snapshot 4/27/2019
    # excludes (data retrieval failure): 'BRK.B', 'DOW', 'FOX', 'FOXA',
    symbols = ['AAPL', 'ABBV', 'ABT', 'ACN', 'AGN', 'AIG', 'ALL', 'AMGN', 'AMZN', 'AXP',
               'BA', 'BAC', 'BIIB', 'BK', 'BKNG', 'BLK', 'BMY', 'C', 'CAT',
               'CELG', 'CHTR', 'CL', 'CMCSA', 'COF', 'COP', 'COST', 'CSCO', 'CVX', 'DHR',
               'DIS', 'DUK', 'DWDP', 'EMR', 'EXC', 'F', 'FB', 'FDX',
               'GD', 'GE', 'GILD', 'GM', 'GOOG', 'GOOGL', 'GS', 'HD', 'HON',
               'IBM', 'INTC', 'JNJ', 'JPM', 'KHC', 'KMI', 'KO', 'LLY', 'LMT', 'LOW',
               'MA', 'MCD', 'MDLZ', 'MDT', 'MET', 'MMM', 'MO', 'MRK', 'MS', 'MSFT',
               'NEE', 'NFLX', 'NKE', 'NVDA', 'ORCL', 'OXY', 'PEP', 'PFE', 'PG', 'PM',
               'PYPL', 'QCOM', 'RTN', 'SBUX', 'SLB', 'SO', 'SPG', 'T', 'TGT', 'TXN',
               'UNH', 'UNP', 'UPS', 'USB', 'UTX', 'V', 'VZ', 'WBA', 'WFC', 'WMT',
               'XOM'
               ]
    data_directory = './data/'
    initial_date = "2014-01-01"
    final_date = "2018-12-31"
    benchmark = 'SPY'
    objective_metrics = ['Sharpe', 'Sortino', 'MAR', 'Info']

    data = populate_data(symbols, data_directory, initial_date, final_date)
    bench_data = populate_data([benchmark], data_directory, initial_date, final_date)

    # --- 2. COMPUTE STUDIES ---
    for ticker in data:
        this_data = data[ticker]  # by reference
        this_data['SMA20'] = this_data.AdjClose.rolling(20).mean()
        this_data['SMA50'] = this_data.AdjClose.rolling(50).mean()
        this_data['SMA200'] = this_data.AdjClose.rolling(200).mean()

    # --- 3. ENUMERATE SEARCH UNIVERSE ---


    # --- 4. CONDUCT WALK-FORWARD ENUMERATION ---




    return
