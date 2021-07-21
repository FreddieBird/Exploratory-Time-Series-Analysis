# Mean reversion tester
# Tests a time series for the likelihood
# that it had mean reverting properties

import statsmodels.tsa.stattools as ts
import warnings
import pandas as pd
import pandas.io.sql as psql
import MySQLdb as mdb
import datetime
import matplotlib
from math import sqrt
from matplotlib import pyplot as plt
from pandas import DataFrame
from numpy import cumsum, log, polyfit, sqrt, std, subtract
from numpy.random import randn
import numpy as np


def hurst(ts):
    """Returns the Hurst Exponent of the time series vector ts"""
    # Create the range of lag values
    lags = range(2, 100)

    ts = ts.tolist()

    # Calculate the array of the variances of the lagged differences
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = polyfit(log(lags), log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0]*2.0


def adfuller(timeseries):
    """Performs the ADF test on a given timeseries"""
    print(ts.adfuller(timeseries))


def main():
    plt.style.use('seaborn-pastel')

    # Connect to the MySQL instance
    db_host = 'localhost'
    db_user = ''
    db_pass = ''
    db_name = ''

    # establish db connection
    con = mdb.connect(db_host, db_user, db_pass, db_name)

    # Select all of the historical close price for each bar of data in D1_price
    sql = """SELECT p.price_date, p.close_price
             FROM symbol as sym
             INNER JOIN D1_price as p
            ON p.symbol_id = sym.id
            WHERE sym.ticker = 'nzdusd'
            ORDER BY p.price_date ASC;"""

    # Create a pandas df from the SQL query
    timeseries = pd.read_sql(sql, con=con)

    # Perform adfuller
    adfuller(timeseries["close_price"])

    # Check Hurst Exponent
    he = hurst(timeseries["close_price"])
    print(f"Hurst: {he}")


if __name__ == '__main__':
    main()
