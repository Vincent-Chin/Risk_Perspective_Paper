import pandas as pd
import numpy as np
import os
import itertools as it
import operator
import math
import datetime as dt
import matplotlib.pyplot as plt
import pathos.multiprocessing as mp

import pull_history as ph
import monte_carlo as mc

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

    if not os.path.exists(data_directory + "performances/"):
        os.mkdir(data_directory + "performances/")

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

    stats['trades'] = len(temp)

    negonly = grouped.copy()
    negonly[negonly > 0] = 0
    stats['semisd'] = np.std(negonly)
    stats['semiannsd'] = stats.semisd * math.sqrt(252)

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

    stats['annInfo'] = benchstats.annInfo
    stats['annsdInfo'] = benchstats.annsdInfo
    stats['InfoRatio'] = stats.annInfo / stats.annsdInfo

    return stats, cum_pl


###########
# EXPERIMENT CODE - MAIN LOGIC
###########
def do_experiment(symbols: list, data_directory: str, initial_date: str, final_date: str, benchmark: str,
                  search_parameters: dict, objective_metrics: list, wf_lookback: int, is_mc: bool):
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

    # --- 2. COMPUTE STUDIES ---
    effect_start = np.busday_offset(initial_date, -200 - wf_lookback - 1)
    data = populate_data(symbols, data_directory, effect_start, final_date)

    # for each stock
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

    # for our benchmark data
    temp = populate_data([benchmark], data_directory, effect_start, final_date)
    bench_data = temp[benchmark]
    bench_returns = (bench_data.AdjClose - bench_data.AdjClose.shift(1)) / bench_data.AdjClose.shift(1)
    bench_returns.columns = ['Return']
    bench_returns.index = pd.to_datetime(bench_returns.index)

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
    test_dates = pd.bdate_range(initial_date, final_date)

    # multiprocessed execution
    wf_record = {}
    pool = mp.Pool()
    for trade_date in test_dates:
        wf_record[trade_date] = pool.apply_async(
            wf_step,
            args=[trade_date, wf_lookback, data_directory, filenames, objective_metrics, bench_returns, is_mc],
            kwds={}, callback=None, error_callback=None)
    pool.close()
    pool.join()

    # tally all results together
    insample_sets = {}
    insample_trades = {}
    outsample_trades = {}
    for tdate in wf_record:
        if len(wf_record[tdate]._value) == 3:
            this_sets, this_in, this_out = wf_record[tdate]._value
            for metric in objective_metrics:
                if metric not in insample_sets:
                    insample_sets = this_sets
                    insample_trades = this_in
                    outsample_trades = this_out
                else:
                    insample_sets[metric] = insample_sets[metric].append(this_sets[metric])
                    insample_trades[metric] = insample_trades[metric].append(this_in[metric])
                    outsample_trades[metric] = outsample_trades[metric].append(this_out[metric])
        else:
            print("Unexpected issue on", tdate, wf_record[tdate]._value)

    # --- 6. GENERATE COMPARATIVE PERFORMANCES FOR ALL OF OUR METRICS ---
    final_perfs = pd.DataFrame()
    insample_pls = {}
    outsample_pls = {}
    for metric in objective_metrics:
        if len(insample_trades[metric]) > 0:
            insample_perf, insample_pl = performance_report(insample_trades[metric].Return, bench_returns)
            insample_perf.index = [(metric, "insample")]
            final_perfs = final_perfs.append(insample_perf)
            insample_pls[metric] = insample_pl
        if len(outsample_trades[metric]) > 0:
            outsample_perf, outsample_pl = performance_report(outsample_trades[metric].Return, bench_returns)
            outsample_perf.index = [(metric, "outsample")]
            final_perfs = final_perfs.append(outsample_perf)
            outsample_pls[metric] = outsample_pl
        insample_pls[metric].index = pd.to_datetime(insample_pls[metric].index)
        outsample_pls[metric].index = pd.to_datetime(outsample_pls[metric].index)

    print(dt.datetime.now(), "Experiment complete.")

    return final_perfs, insample_pls, outsample_pls, bench_returns


def wf_step(trade_date, wf_lookback, data_directory, filenames, objective_metrics, bench_returns, is_mc,
            mc_iterations=50):
    print(dt.datetime.now(), "Conducting experiment on", trade_date, "; monte carlo:", is_mc)

    insample_sets = {}
    insample_trades = {}
    outsample_trades = {}
    for metric in objective_metrics:
        insample_sets[metric] = pd.DataFrame(index=[trade_date], columns=['Combo'])
        insample_trades[metric] = pd.DataFrame()
        outsample_trades[metric] = pd.DataFrame()

    first_date = np.busday_offset(trade_date.date(), -wf_lookback - 1)  # first date of calibration period
    last_date = np.busday_offset(trade_date.date(), - 1)  # last date of calibration period

    # go through each combination and aggregate the performances
    if not is_mc:
        perf_file = data_directory + "performances/" + str(trade_date.date()) + ".csv"
    else:
        perf_file = data_directory + "performances/mc_" + str(trade_date.date()) + ".csv"

    if os.path.exists(perf_file):
        performances = pd.read_csv(perf_file, index_col=0)
    else:
        # we have two modes here, traditional, and monte carlo.  in traditional we just do a straight backtest over
        # the test period for each combo's performance (up to the trade date), and then record the performance.
        #
        # monte carlo differs in that we're first going to generate n samples of dates, and then extract signals
        # corresponding to those dates and run a performance for each of those combos.  we then average all of those
        # performances to determine the performance for the overall combo.
        mc_samples = []
        if is_mc:
            sampling_frame = pd.DataFrame(index=pd.bdate_range(first_date, last_date))
            for n in range(0, mc_iterations):
                mc_samples.append(mc.montecarlo_sample(sampling_frame, 5))

        performances = pd.DataFrame()
        for combo in filenames:
            this_signals = pd.read_csv(filenames[combo], index_col=0, parse_dates=True)
            if len(this_signals) > 0:
                this_signals = this_signals[(this_signals.index.date >= first_date)
                                            & (this_signals.index.date <= last_date)]
                this_benchmark = bench_returns[(bench_returns.index.date >= first_date)
                                               & (bench_returns.index.date <= last_date)]

                if len(this_signals) > 0:
                    if not is_mc:
                        # if doing traditional execution, just pull out trades from the test period
                        this_performance, this_cumpl = performance_report(this_signals.Return, this_benchmark)
                        this_performance.index = [str(combo)]
                        performances = performances.append(this_performance)
                    else:
                        # if doing monte carlo execution, we need to repeatedly sample trades and then average all of
                        # the performances.
                        sample_performances = pd.DataFrame()
                        for this_sample in mc_samples:
                            sample_signals = this_signals.loc[this_signals.index.isin(this_sample.index)]
                            if len(sample_signals) > 0:
                                this_performance, this_cumpl = performance_report(sample_signals.Return, this_benchmark)
                                sample_performances = sample_performances.append(this_performance, ignore_index=True)

                        if 'annPL' in sample_performances.columns:
                            avg_performance = sample_performances.mean()
                            avg_performance.name = str(combo)
                            # recompute the derived stats using the PL info, rather than using the raw average.
                            avg_performance['Sharpe'] = avg_performance.annPL / avg_performance.annSD
                            avg_performance['Sortino'] = avg_performance.annPL / avg_performance.semiannsd
                            avg_performance['MAR'] = avg_performance.annPL / abs(avg_performance.MaxDD)
                            avg_performance['InfoRatio'] = avg_performance.annInfo / avg_performance.annsdInfo
                            performances = performances.append(pd.DataFrame(avg_performance).transpose())

        performances.to_csv(perf_file)

    # sort by our objective metrics.  ignore results with <30 trades or with infinite ratios or negative PL.
    for metric in objective_metrics:
        subset_perfs = performances.sort_values([metric], ascending=False).copy()
        subset_perfs = subset_perfs[(subset_perfs.trades >= 30)
                                    & (np.isinf(subset_perfs[metric]) == False)
                                    & (subset_perfs.rawPL > 0)]
        if len(subset_perfs) >= 1:
            best_perf = subset_perfs.iloc[0].copy()
            combo = best_perf.name
            filename = data_directory + "signals/" + str(combo).replace(" ", "").replace(",", "_") \
                .replace("'", "").replace(":", "-").replace("<", "L").replace(">", "G").replace("=", "E") + ".csv"
            best_trades = pd.read_csv(filename, index_col=0)

            insample_sets[metric].loc[trade_date, 'Combo'] = best_perf.name
            insample_trades[metric] = insample_trades[metric].append(
                best_trades[pd.to_datetime(best_trades.index).date == last_date])
            outsample_trades[metric] = outsample_trades[metric].append(
                best_trades[pd.to_datetime(best_trades.index).date == trade_date.date()])

            # write out identified information for manual verification purposes - doesn't work with multiprocessing
            # insample_sets[metric].to_csv(data_directory + "performances/" + metric + "_sets.csv")
            # insample_trades[metric].to_csv(data_directory + "performances/" + metric + "_intrades.csv")
            # outsample_trades[metric].to_csv(data_directory + "performances/" + metric + "_outtrades.csv")

    return insample_sets, insample_trades, outsample_trades


def plot(objective_metrics, bench_returns, insample_pls, outsample_pls, is_mc):
    # combine data into a single plottable dataframe
    date_range = None
    combined_frame = pd.DataFrame()
    for metric in objective_metrics:
        if len(combined_frame) == 0:
            date_range = pd.bdate_range(insample_pls[metric].index[0],
                                        outsample_pls[metric].index[len(outsample_pls[metric]) - 1])
            combined_frame = pd.DataFrame(index=date_range)

        combined_frame[(metric, "insample")] = insample_pls[metric]
        combined_frame[(metric, "outsample")] = outsample_pls[metric]

    subset_benchmark = bench_returns[(bench_returns.index >= date_range[0])
                                     & (bench_returns.index <= date_range[len(date_range) - 1])]
    combined_frame['benchmark'] = np.cumsum(subset_benchmark)
    combined_frame.ffill(inplace=True)

    # plot
    ax = combined_frame.plot(colormap='Set1')
    ax.xaxis.grid()
    ax.yaxis.grid()
    ax.set_facecolor('white')
    if not is_mc:
        plt.title('Cumulative Returns')
    else:
        plt.title('Cumulative Returns - Monte Carlo')
    plt.show()

    return


###########
# MAIN() - FUNCTIONAL SCAFFOLDING
###########
def main():

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
    objective_metrics = ['Sharpe', 'Sortino', 'MAR', 'InfoRatio']
    wf_lookback = 100
    is_mc = False
    mc_iterations = 50

    # --- 2. Conduct traditional experiment.  See do_experiment() code for step details.
    final_perfs, insample_pls, outsample_pls, bench_returns = do_experiment(
        symbols, data_directory, initial_date, final_date, benchmark, search_parameters, objective_metrics,
        wf_lookback, is_mc=False)
    plot(objective_metrics, bench_returns, insample_pls, outsample_pls, False)
    print(final_perfs)

    # --- 3. Conduct monte carlo experiment.
    mc_final_perfs, mc_insample_pls, mc_outsample_pls, mc_bench_returns = do_experiment(
        symbols, data_directory, initial_date, final_date, benchmark, search_parameters, objective_metrics,
        wf_lookback, is_mc=True)
    plot(objective_metrics, bench_returns, mc_insample_pls, mc_outsample_pls, True)
    print(mc_final_perfs)

    return


###########
# DEBUG HELPERS
###########
def debug_scratch(filename, bench_returns):
    signals = pd.read_csv(filename, index_col=0)
    rets = signals.Return
    perf, cum_pls = performance_report(rets, bench_returns)
    return


def debug_performance(data_directory, objective_metrics, bench_returns):
    insample_trades = {}
    outsample_trades = {}
    insample_pls = {}
    outsample_pls = {}

    final_perfs = pd.DataFrame()
    for metric in objective_metrics:
        filename = data_directory + "performances/" + metric + "_intrades.csv"
        insample_trades[metric] = pd.read_csv(filename, index_col=0, parse_dates=True)
        if len(insample_trades[metric]) > 0:
            insample_perf, insample_pl = performance_report(insample_trades[metric].Return, bench_returns)
            insample_perf.index = [(metric, "insample")]
            final_perfs = final_perfs.append(insample_perf)
            insample_pls[metric] = insample_pl

        filename = data_directory + "performances/" + metric + "_outtrades.csv"
        outsample_trades[metric] = pd.read_csv(filename, index_col=0, parse_dates=True)
        if len(outsample_trades[metric]) > 0:
            outsample_perf, outsample_pl = performance_report(outsample_trades[metric].Return, bench_returns)
            outsample_perf.index = [(metric, "outsample")]
            final_perfs = final_perfs.append(outsample_perf)
            outsample_pls[metric] = outsample_pl
    return
