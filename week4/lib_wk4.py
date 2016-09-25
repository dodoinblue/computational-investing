#!/usr/bin/env python

# QSTK Imports
import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da
#import QSTK.qstkstudy.EventProfiler as ep

# Third Party Imports
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy

import sys
sys.path.insert(0, '../week3')

import lib

def getData(startdate, enddate, symbols, keys):
    d_data = lib.getDataWithAllKeys(startdate, enddate, symbols, keys)

    # fill NaNs
    for key in keys:
        d_data[key] = d_data[key].fillna(method = 'ffill')
        d_data[key] = d_data[key].fillna(method = 'bfill')
        d_data[key] = d_data[key].fillna(1.0)

    return d_data

def find_events_down_5_percent(ls_symbols, d_data):
    df_close = d_data['close']
    ts_market = df_close['SPY']

    df_events = copy.deepcopy(df_close)
    df_events = df_events * np.NAN

    ldt_timestamps = df_close.index

    for s_sym in ls_symbols:
        for i in range(1, len(ldt_timestamps)): # for each day
            # Calculating the returns for this timestamp
            f_symprice_today = df_close[s_sym].ix[ldt_timestamps[i]]
            f_symprice_yest = df_close[s_sym].ix[ldt_timestamps[i - 1]]
            f_marketprice_today = ts_market.ix[ldt_timestamps[i]]
            f_marketprice_yest = ts_market.ix[ldt_timestamps[i - 1]]
            f_symreturn_today = (f_symprice_today / f_symprice_yest) - 1
            f_marketreturn_today = (f_marketprice_today / f_marketprice_yest) - 1

            # Event is found if the symbol is down more then 3% while the
            # market is up more then 2%
            if f_symreturn_today <= -0.03 and f_marketreturn_today >= 0.02:
                 df_events[s_sym][i] = 1

    return df_events


def find_events_below_thres(ls_symbols, d_data, thres):
    df_close = d_data['actual_close']
    df_events = copy.deepcopy(df_close)
    df_events = df_events * np.NAN

    ldt_timestamps = df_close.index

    for s_sym in ls_symbols:
        for i in range(1, len(df_close[s_sym])):
            f_symprice_today = df_close[s_sym].ix[ldt_timestamps[i]]
            f_symprice_yest = df_close[s_sym].ix[ldt_timestamps[i - 1]]

            if f_symprice_yest >= thres and f_symprice_today < thres:
                df_events[s_sym][i] = 1

    return df_events

def eventprofiler(df_events_arg, d_data, i_lookback=20, i_lookforward=20,
                s_filename='study', b_market_neutral=True, b_errorbars=True,
                s_market_sym='SPY'):
    ''' Event Profiler for an event matix'''
    df_close = d_data['close'].copy()
    df_rets = df_close.copy()

    # Do not modify the original event dataframe.
    df_events = df_events_arg.copy()
    tsu.returnize0(df_rets.values)

    if b_market_neutral == True:
        df_rets = df_rets - df_rets[s_market_sym]
        del df_rets[s_market_sym]
        del df_events[s_market_sym]

    df_close = df_close.reindex(columns=df_events.columns)

    # Removing the starting and the end events
    df_events.values[0:i_lookback, :] = np.NaN
    df_events.values[-i_lookforward:, :] = np.NaN

    # Number of events
    i_no_events = int(np.logical_not(np.isnan(df_events.values)).sum())
    assert i_no_events > 0, "Zero events in the event matrix"
    na_event_rets = "False"

    # Looking for the events and pushing them to a matrix
    for i, s_sym in enumerate(df_events.columns):
        for j, dt_date in enumerate(df_events.index):
            if df_events[s_sym][dt_date] == 1:
                na_ret = df_rets[s_sym][j - i_lookback:j + 1 + i_lookforward]
                if type(na_event_rets) == type(""):
                    na_event_rets = na_ret
                else:
                    na_event_rets = np.vstack((na_event_rets, na_ret))

    if len(na_event_rets.shape) == 1:
        na_event_rets = np.expand_dims(na_event_rets, axis=0)

    # Computing daily rets and retuns
    na_event_rets = np.cumprod(na_event_rets + 1, axis=1)
    na_event_rets = (na_event_rets.T / na_event_rets[:, i_lookback]).T

    # Study Params
    na_mean = np.mean(na_event_rets, axis=0)
    na_std = np.std(na_event_rets, axis=0)
    li_time = range(-i_lookback, i_lookforward + 1)

    # Plotting the chart
    plt.clf()
    plt.axhline(y=1.0, xmin=-i_lookback, xmax=i_lookforward, color='k')
    if b_errorbars == True:
        plt.errorbar(li_time[i_lookback:], na_mean[i_lookback:],
                    yerr=na_std[i_lookback:], ecolor='#AAAAFF',
                    alpha=0.7)
    plt.plot(li_time, na_mean, linewidth=3, label='mean', color='b')
    plt.xlim(-i_lookback - 1, i_lookforward + 1)
    if b_market_neutral == True:
        plt.title('Market Relative mean return of ' +\
                str(i_no_events) + ' events')
        print 'Number of events count in report: ', str(i_no_events)
    else:
        plt.title('Mean return of ' + str(i_no_events) + ' events')
    plt.xlabel('Days')
    plt.ylabel('Cumulative Returns')
    plt.savefig(s_filename, format='pdf')


def test():
    startdate = dt.datetime(2008,1,1)
    enddate = dt.datetime(2009,12,31)
    symbol_list = 'sp5002008'

    d_data = getData(startdate, enddate, symbol_list)

    return d_data[key].values

if __name__ == '__main__':
    start_year = int(sys.argv[1])
    end_year = int(sys.argv[2])
    symbol_list = sys.argv[3]
    thres = int(sys.argv[4])

    print 'start_year: ', start_year
    print 'end_year: ', end_year
    print 'symbol: ', symbol_list
    print 'thres: ', thres

    startdate = dt.datetime(start_year,1,1)
    enddate = dt.datetime(end_year,12,31)
    # symbol_list = 'sp5002012'

    dataobj = da.DataAccess('Yahoo')
    ls_symbols = dataobj.get_symbols_from_list(symbol_list)
    ls_symbols.append('SPY')
    keys = ['close', 'actual_close']

    d_data = getData(startdate, enddate, ls_symbols, keys)

    df_events = find_events_below_thres(ls_symbols, d_data, thres)

    # print df_events

    print 'events in SPY: ', (df_events['SPY'].values == 1).sum()

    print 'Total events: ', (df_events.values[:,:-1] == 1).sum()
    i_no_events = int(np.logical_not(np.isnan(df_events.values)).sum())
    print 'Actual events: ', i_no_events

    eventprofiler(df_events, d_data, i_lookback=20, i_lookforward=20,
                s_filename='MyEventStudy.pdf', b_market_neutral=True, b_errorbars=True,
                s_market_sym='SPY')
