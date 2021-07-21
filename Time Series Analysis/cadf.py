# Cointegrated Augmented Dickey-Fuller test

import MySQLdb as mdb
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import pprint
import statsmodels.tsa.stattools as ts
import statsmodels.formula.api as sm
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

ts1_name = ""
ts2_name = ""


def plot_price_series(ts1, ts2):
    """Plots two price series on the same chart.
    Allows us to visually inspect whether any cointegration may be likely."""

    fmt_year = mdates.MonthLocator(interval=12)  # every month
    fig, ax = plt.subplots()
    ax.plot(ts1["price_date"], ts1["close_price"], label=ts1_name)
    ax.plot(ts2["price_date"], ts2["close_price"], label=ts2_name)
    ax.xaxis.set_major_locator(fmt_year)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.set_xlim(datetime.datetime(2009, 8, 1), datetime.datetime(2020, 7, 31))
    ax.grid(True)
    fig.autofmt_xdate()

    plt.xlabel('Month/Year')
    plt.ylabel('Price ($)')
    plt.title(f"{ts1_name} and {ts2_name} Daily Prices")
    plt.legend()
    plt.show()


def plot_scatter_series(ts1, ts2):
    """Plots a scatter plot of the two prices.
    Allows us to visually inspect whether a linear
    relationship exists between the two series
    and thus whether it is a good candidate for OLS and ADF."""

    plt.xlabel(f"{ts1_name} Price ($)")
    plt.ylabel(f"{ts2_name} Price ($)")
    plt.title(f"{ts1_name} and {ts2_name} Price Scatterplot")
    plt.scatter(ts1, ts2)
    plt.show()


def plot_residuals(df):
    """Plots the residual values from the fitted linear model
    of the two price series. """

    print(df.head())

    fmt_years = mdates.MonthLocator(interval=12)  # every year
    fig, ax = plt.subplots()
    ax.plot(df.iloc[:, 0], df["res"], label="Residuals")
    ax.xaxis.set_major_locator(fmt_years)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_xlim(datetime.datetime(2009, 8, 1), datetime.datetime(2020, 7, 31))
    ax.grid(True)
    fig.autofmt_xdate()

    plt.xlabel("Month/Year")
    plt.ylabel("Price ($)")
    plt.title("Residual Plot")
    plt.legend()

    plt.plot(df["res"])
    plt.show()


def main():
    plt.style.use('seaborn-pastel')

    global ts1_name
    global ts2_name

    ts1_name = "audusd"
    ts2_name = "usdcad"

    # Connect to the MySQL instance
    db_host = 'localhost'
    db_user = ''
    db_pass = ''
    db_name = ''

    # establish db connection
    con = mdb.connect(db_host, db_user, db_pass, db_name)

    # Select all of the historical close price for each bar of data in D1_price
    sql1 = f"""SELECT p.price_date, p.close_price
             FROM symbol as sym
             INNER JOIN D1_price as p
            ON p.symbol_id = sym.id
            WHERE sym.ticker = '{ts1_name}'
            ORDER BY p.price_date ASC;"""

    # Create a pandas df from the SQL query
    timeseries1 = pd.read_sql(sql1, con=con)

    # Select all of the historical close price for each bar of data in D1_price
    sql2 = f"""SELECT p.price_date, p.close_price
             FROM symbol as sym
             INNER JOIN D1_price as p
            ON p.symbol_id = sym.id
            WHERE sym.ticker = '{ts2_name}'
            ORDER BY p.price_date ASC;"""

    # Create a pandas df from the SQL query
    timeseries2 = pd.read_sql(sql2, con=con)

    # print(timeseries1.tail())
    # print(timeseries2.tail())
    # print(len(timeseries1["close_price"]))
    # print(len(timeseries2["close_price"]))

    # Join both price series and drop NaN values
    # - allows scatter and OLS analysis to work
    df = pd.concat([timeseries1, timeseries2], axis=1, join="inner")
    df.dropna()
    print(df.head())
    ts1_close_price = df.iloc[:, 1]
    ts2_close_price = df.iloc[:, 3]

    plot_price_series(timeseries1, timeseries2)
    plot_scatter_series(ts1_close_price, ts2_close_price)

    # Perform the Linear Regression between the two time series
    # and calculate the optimal hedge ratio 'beta'
    #timeseries1['intercept'] = 1
    X = ts1_close_price
    X = sm.add_constant(X)
    y = ts2_close_price
    result = sm.OLS(y, X).fit()
    print(result.summary())
    beta_hr = result.params[1]

    # Calculate the residuals of the linear combination
    df["res"] = ts2_close_price - beta_hr*ts1_close_price

    # Plot the residuals
    plot_residuals(df)

    # Calculate and output the CADF test on the residuals
    cadf = ts.adfuller(df["res"])
    pprint.pprint(cadf)


if __name__ == '__main__':
    main()
