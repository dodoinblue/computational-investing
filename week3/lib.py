#!/usr/bin/env python

# QSTK Imports
import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da

# Third Party Imports
import datetime as dt
#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def output(str):
    print 'Output: ', str

def getAdjustedClose(startdate, enddate, symbols):
    closeKey = 'close'
    d_data = getDataWithAllKeys(startdate, enddate, symbols, [closeKey])
    return d_data[closeKey].values


def getDataWithAllKeys(startdate, enddate, symbols, keys):
    c_dataobj = da.DataAccess('Yahoo')
    dt_timeofday = dt.timedelta(hours=16)
    ldt_timestamps = du.getNYSEdays(startdate, enddate, dt_timeofday)
    ldf_data = c_dataobj.get_data(ldt_timestamps, symbols, keys)
    d_data = dict(zip(keys, ldf_data))
    return d_data

def simulate(start_date, end_date, symbols, alloc):

	na_price = getAdjustedClose(start_date, end_date, symbols)
	na_price_norm = na_price / na_price[0]
	na_daily_vals = na_price_norm.dot(alloc)
	na_daily_rets = na_daily_vals[1:] / na_daily_vals[:-1] - 1
	na_daily_rets = np.append([.0], na_daily_rets)

	k = np.sqrt(252)
	std_daily_ret = na_daily_rets.std()
	avg_daily_ret = na_daily_rets.mean()
	sharp_ratio = k * avg_daily_ret / std_daily_ret

	return std_daily_ret, avg_daily_ret, sharp_ratio, na_daily_vals[-1]

def portfolio_optimizer(start_date, end_date, symbols):
	alloc_range = np.arange(0.,1.1,.1)
	combinations = [[round(x1,1),round(x2,1),round(x3,1),round(x4,1)]
					for x1 in alloc_range for x2 in alloc_range for x3 in alloc_range for x4 in alloc_range
					if x1+x2+x3+x4 == 1]

	ls_performance = [(com, simulate(start_date, end_date, symbols, com)) for com in combinations]
	performance = {tuple(key): value for (key, value) in ls_performance}
	best_alloc = max(performance, key = lambda x: performance.get(x)[2])
	best_result = performance[best_alloc]

	print 'Start Data: ' + start_date.strftime('%B %d, %Y')
	print 'End Data: ' + end_date.strftime('%B %d, %Y')
	print 'Symbols: ' + str(symbols)
	print 'Optimal Allocations: ' + str(list(best_alloc))
	print 'Sharp Ratio: %.11f'%best_result[2]
	print 'Volatility (stdev of daily returns): %.13f'%best_result[0]
	print 'Average Daily Return: %.15f'%best_result[1]
	print 'Cumulative Return: %.12f'%best_result[3]


if __name__ == '__main__':
	portfolio_optimizer(dt.datetime(2010,1,1), dt.datetime(2010,12,31),  ['C', 'GS', 'IBM', 'HNZ'])
