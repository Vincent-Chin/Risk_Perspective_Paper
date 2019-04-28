import pandas as pd
import numpy as np
import os
import itertools as it
import operator
import math

import pull_history as ph

pd.options.display.max_rows = 999
pd.options.display.max_columns = 999


###########
# HELPER FUNCTIONS
###########
def populate_data(symbols, data_directory, initial_date, final_date):

    if not os.path.exists(data_directory):
        os.mkdir(data_directory)

    if not os.path.exists(data_directory + "signals/"):
        os.mkdir(data_directory + "signals/")

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


def dict_product(dicts):
    return (dict(zip(dicts, x)) for x in it.product(*dicts.values()))


def generate_signals(combo, data):
    allowed_operators = {
        ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
    }

    all_signals = pd.DataFrame()
    for symbol in data:
        signals = data[symbol].copy()
        signals['Symbol'] = symbol
        signals['doAppend'] = (allowed_operators[combo['MA20_Mode']](signals['x20'], combo['MA20_Std'])
                                 & allowed_operators[combo['MA200_Mode']](signals['x200'], combo['MA200_Std']))
        all_signals = all_signals.append(signals[signals.doAppend], sort=True)

    return all_signals


def performance_report(rets, bench_returns):

    # group the returns by date.  we're going to assume equal-weighting in the returns.
    temp = rets.reset_index()
    grouped = temp.groupby(['Timestamp']).mean()

    # generate descriptive statistics
    stats = pd.DataFrame()
    stats['rawPL'] = np.sum(grouped)
    stats['sdPL'] = np.std(grouped)
    stats['days'] = np.busday_count(pd.Timestamp(grouped.index[0]).date(),
                                    pd.Timestamp(grouped.index[len(grouped) - 1]).date())
    stats['annPL'] = (stats.rawPL / stats.days) * 252
    stats['annSD'] = stats.sdPL * math.sqrt(252)

    negonly = grouped.copy()
    negonly[negonly > 0] = 0
    stats['semiannsd'] = np.std(negonly) * math.sqrt(252)

    cum_pl = np.cumsum(grouped)
    dd = cum_pl - cum_pl.cummax()
    stats['MaxDD'] = np.min(dd)

    # ratio computation
    stats['Sharpe'] = stats.annPL / stats.annSD
    stats['Sortino'] = stats.annPL / stats.semiannsd
    stats['MAR'] = stats.annPL / abs(stats.MaxDD)

    # info-ratio computations
    vsBench = grouped.Return - bench_returns
    benchstats = pd.Series()
    benchstats['rawInfo'] = np.sum(vsBench)
    benchstats['sdInfo'] = np.std(vsBench)
    benchstats['annInfo'] = (benchstats.rawInfo / stats.days) * 252
    benchstats['annsdInfo'] = benchstats.sdInfo * math.sqrt(252)
    stats['Info'] = benchstats.annInfo / benchstats.annsdInfo

    return stats, cum_pl


###########
# MAIN LOGIC
###########
def main():
    # PROCESS
    # -------
    # 1. stage data - pull from csvs
    # 2. compute study values (moving averages - 20 period, 50 period, 200 period)
    # 3. enumerate search universe
    # 4. generate signals for each enumeration of the search universe
    #
    # 5. for each search metric:
    #   a. for each date of testing period:
    #       1. isolate data bounds using lookback
    #       3. apply filtering constraints for each permutation of the search universe
    #       4. cache signal performance on a returns basis
    #       5. choose best performer on day n-1 for active signals on day n, by search metric
    #       6. aggregate signals onto a outsample and insample signals file for the search metric
    #
    # 6. for each signals file:
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
    search_parameters = {
        'MA20_Std': np.arange(-2.5, 2.5, 0.5),
        'MA200_Std': np.arange(-2.5, 2.5, 0.5),
        'MA20_Mode': ('>', '<'),
        'MA200_Mode': ('>', '<'),
    }

    # --- 2. COMPUTE STUDIES ---
    effect_start = np.busday_offset(initial_date, -200)
    data = populate_data(symbols, data_directory, effect_start, final_date)

    for ticker in data:
        this_data = data[ticker]  # by reference
        # these require a negative-1 shift because we only know up to the prior close.
        this_data['MA20'] = this_data.AdjClose.rolling(20).mean().shift(1)
        this_data['SD20'] = this_data.AdjClose.rolling(20).std().shift(1)
        this_data['MA200'] = this_data.AdjClose.rolling(200).mean().shift(1)
        this_data['SD200'] = this_data.AdjClose.rolling(200).std().shift(1)
        this_data['x20'] = (this_data.AdjClose.shift(1) - this_data.MA20) / this_data.SD20
        this_data['x200'] = (this_data.AdjClose.shift(1) - this_data.MA200) / this_data.SD200
        # this is used for comparison purposes so no adjustment is needed.
        this_data['Return'] = (this_data.AdjClose - this_data.AdjClose.shift(1)) / this_data.AdjClose.shift(1)

    temp = populate_data([benchmark], data_directory, effect_start, final_date)
    bench_data = temp[benchmark]
    bench_returns = (bench_data.AdjClose - bench_data.AdjClose.shift(1)) / bench_data.AdjClose.shift(1)
    bench_returns.columns = ['Return']

    # --- 3. ENUMERATE SEARCH UNIVERSE ---
    combos = dict_product(search_parameters)
    search_universe = []
    for combo in list(combos):
        this_combo = dict(combo)
        search_universe.append(this_combo)

    # --- 4. GENERATE RAW SIGNAL SET ---
    filenames = {}
    for combo in search_universe:
        filename = data_directory + "signals/" + str(combo).replace(" ", "").replace(",", "_") \
            .replace("'", "").replace(":", "-").replace("<", "L").replace(">", "G").replace("=", "E") + ".csv"

        if os.path.exists(filename):
            filenames[str(combo)] = filename
        else:
            signals = generate_signals(combo, data)
            signals.to_csv(filename)
            filenames[str(combo)] = filename

    # --- 5. CONDUCT WALK-FORWARD ENUMERATION ---
    objective_metrics = ['Sharpe', 'Sortino', 'MAR', 'Info']


    return


def debug_scratch(filename, bench_returns):
    signals = pd.read_csv(filename, index_col=0)
    rets = signals.Return
    perf, cum_pls = performance_report(rets, bench_returns)
    return