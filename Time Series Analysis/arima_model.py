"""
File containing ARIMA model functionality:
    1) finds best ARIMA model based on out-of-sample RMSE
    2) finds best ARIMA model based on in-sample AIC
    3) forecasts future
"""

import warnings
import pandas as pd
import pandas.io.sql as psql
import MySQLdb as mdb
import datetime
import matplotlib
from math import sqrt
from matplotlib import pyplot as plt
from pandas import DataFrame
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_model import ARMAResults
from sklearn.metrics import mean_squared_error, matthews_corrcoef

# splits data into train, validation and testing


def prepare_train_val_test(X, split=0.5):
    # prepare training dataset
    train_size = int(len(X)*split)
    train, val, test = X[0:train_size], X[train_size:train_size +
                                          int(train_size/2)], X[train_size+int(train_size/2):]
    history = [x for x in train]
    return train, val, test, history


# evaluate an ARIMA model for a given order (p,d,q)
# using out-of-sample RMSE
def evaluate_arima_model_rmse(X, arima_order):

    train, val, test, history = prepare_train_test(X)

    # made predictions
    predictions = list()
    for t in range(len(val)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test.iloc[t])

    # calculate out of sample error
    rmse = sqrt(mean_squared_error(val, predictions))

    return rmse


# evaluate combinations of p, d and q for an ARIMA model
# using out-of-sample RMSE
def evaluate_models_rmse(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    rmse = evaluate_arima_model_rmse(dataset, order)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
                    print('ARIMA%s RMSE=%.3f' % (order, rmse))
                except:
                    continue

    # summarise best model
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))
    train, val, test, history = prepare_train_val_test(dataset)
    model = ARIMA(history, order=best_cfg)
    model_fit = model.fit()
    print(model_fit.summary())
    residuals = DataFrame(model_fit.resid)
    plot_acf(residuals)
    residuals.plot(kind='kde')
    print(residuals.describe())
    plt.show()

    return model_order


# evaluate an ARIMA model for a given order (p,d,q)
# using in-sample AIC
def evaluate_arima_model_aic(X, arima_order):

    train, val, test, history = prepare_train_val_test(X)

    model = ARIMA(history, order=arima_order)
    model_fit = model.fit()
    aic = model_fit.aic

    return aic


# evaluate combinations of p, d and q for an ARIMA model
# using in-sample AIC
def evaluate_models_aic(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                aic = evaluate_arima_model_aic(dataset, order)
                if aic < best_score:
                    best_score, best_cfg = aic, order
                print('ARIMA%s AIC=%.3f' % (order, aic))

    # summarise best model
    print('Best ARIMA%s AIC=%.3f' % (best_cfg, best_score))
    train, val, test, history = prepare_train_val_test(dataset)
    model = ARIMA(history, order=best_cfg)
    model_fit = model.fit()
    print(model_fit.summary())
    residuals = DataFrame(model_fit.resid)
    plot_acf(residuals)
    residuals.plot(kind='kde')
    print(residuals.describe())
    plt.show()

    return best_cfg


# forecasts model on the test set of data
# and returns regression and classification metrics
def forecast_model(dataset, model_order):
    dataset = dataset.astype('float32')
    #train, val, test, history = prepare_train_val_test(dataset)

    #train_val = pd.concat([train, val], axis=1)
    #history = [x for x in train_val]
    train_size = int(len(dataset)*0.75)
    train, test = dataset[0:train_size], dataset[train_size:]
    history = [x for x in train]

    # made predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=model_order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test.iloc[t])
        print("Predicted=%f, Expected=%f" % (yhat, test.iloc[t]), end='\r')

    x_axis = [x for x in range(len(test))]
    plt.plot(x_axis, test, label='Expected')
    plt.plot(x_axis, predictions, label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # calculate out of sample regression error
    rmse = sqrt(mean_squared_error(test, predictions))
    print("Test RMSE: ", rmse)

    # calculate out of sample classification error
    test_signal = []
    predictions_signal = []
    for i in range(len(predictions)-1):
        if predictions[i+1] > predictions[i]:
            predictions_signal.append(1)
        else:
            predictions_signal.append(-1)

        if test.iloc[i+1] > test.iloc[i]:
            test_signal.append(1)
        else:
            test_signal.append(-1)

    print("Test MCC: ", matthews_corrcoef(test_signal,
                                          predictions_signal))


if __name__ == "__main__":

    plt.style.use('seaborn-pastel')

    # Connect to the MySQL instance
    db_host = 'localhost'
    db_user = ''
    db_pass = ''
    db_name = ''

    # establish db connection
    con = mdb.connect(db_host, db_user, db_pass, db_name)

    # Select all of the historical close price for each bar of data in H1_price
    sql = """SELECT d1p.price_date, d1p.close_price
             FROM symbol as sym
             INNER JOIN D1_price as d1p
            ON d1p.symbol_id = sym.id
            WHERE sym.ticker = 'eurusd'
            ORDER BY d1p.price_date ASC;"""

    # Create a pandas df from the SQL query
    eurusd = pd.read_sql(sql, con=con)

    # evaluate parameters
    p_values = range(3, 10)
    d_values = range(1, 3)
    q_values = range(3, 10)

    warnings.filterwarnings("ignore")

    # find best arima model
    #model_order = evaluate_models_rmse(eurusd["close_price"], p_values, d_values, q_values)
    model_order = evaluate_models_aic(eurusd["close_price"], p_values, d_values, q_values)

    # forecast on the test set of data and measure performance
    forecast_model(eurusd["close_price"], model_order)
