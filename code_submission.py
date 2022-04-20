import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import datetime as dt
import itertools

from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

def read_data(csv_file: str) -> pd.DataFrame:
    """
    Read csv from a given path
    :param csv_file: str path to .csv file
    :return: dataframe, dataframe summary
    """
    return pd.read_csv(csv_file), pd.read_csv(csv_file).describe()

def test_stationarity(timeseries: pd.DataFrame) -> True:
    """
    Test whether the given time-series, y is stationary or not using the ADF test. Also plots the rolling statistics to examine stationarity
    :param timeseries: date-indexed dataframe with one column or pandas Series object
    :return: True if stationary
    """
    # Rolling statistics
    rolmean = timeseries.rolling(7).mean()
    rolstd = timeseries.rolling(7).std()

    # Plot rolling statistics:
    plt.plot(timeseries, color='blue', label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # Perform Dickey-Fuller test:
    # print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    # print(dfoutput)

    if dfoutput['p-value'] < 0.05:
        return True
    else:
        return False

def train_valid_test_split(X: pd.DataFrame):
    """
    Split the dataset into 80/20 for train and test to evaluate the model
    :param X: date indexed dataframe
    :return: train and test dataframes
    """
    train, test = train_test_split(X, test_size=0.2, random_state=1, shuffle=False)

    return train, test


def get_pred_from_sarimax(df: pd.DataFrame, is_exog: True, df_exog: pd.DataFrame, forecast_steps: int):
    """
    Implement sarima with parameters selected through ACF/PACF plots and return predictions over a given forecast length
    :param df: data indexed pandas dataframe
    :param is_exog: flag for exogenous variables
    :param df_exog: exog dataframe
    :param forecast_steps: desired forecasted values
    :return: model object, predictions dataframe and confidence interval dataframe
    """
    if is_exog == True:
        model = sm.tsa.statespace.SARIMAX(df, exog=df_exog[df.index.min():df.index.max()],
                                          order=(3, 0, 3),
                                          seasonal_order=(0, 1, 0, 7),
                                          enforce_stationarity=False,
                                          freq='D',
                                          enforce_invertibility=False).fit()

        start_index = df.index.max() + dt.timedelta(1)
        end_index = start_index + dt.timedelta(days=forecast_steps - 1)

        predictions = pd.DataFrame(model.predict(start=start_index, end=end_index, exog=df_exog[start_index:end_index]))
        predictions.columns = ['Counts']
        predictions[predictions < 0] = 0

        get_pred = model.get_prediction(start=start_index, end=end_index, dynamic=False,
                                        exog=df_exog[start_index:end_index])
        predictions_ci = get_pred.conf_int(alpha=0.05)
        predictions_ci[predictions_ci < 0] = 0
    else:
        model = sm.tsa.statespace.SARIMAX(df,
                                          order=(3, 1, 3),
                                          seasonal_order=(0, 1, 0, 7),
                                          enforce_stationarity=False,
                                          freq='D',
                                          enforce_invertibility=False).fit()

        start_index = df.index.max() + dt.timedelta(1)
        end_index = start_index + dt.timedelta(days=forecast_steps - 1)

        predictions = pd.DataFrame(model.predict(start=start_index, end=end_index))
        predictions.columns = ['Counts']
        predictions[predictions < 0] = 0

        get_pred = model.get_prediction(start=start_index, end=end_index, dynamic=False)
        predictions_ci = get_pred.conf_int(alpha=0.05)
        predictions_ci[predictions_ci < 0] = 0

    return model, predictions, predictions_ci


def get_pred_from_sarimax_tuned(df: pd.DataFrame, is_exog: True, df_exog: pd.DataFrame, forecast_steps: int):
    """
    Implement grid search for optimal sarima and return predictions over a given forecast length
    :param df: data indexed pandas dataframe
    :param is_exog: flag for exogenous variables
    :param df_exog: exog dataframe
    :param forecast_steps: desired forecasted values
    :return: model object, predictions dataframe and confidence interval dataframe
    """
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 7) for x in list(itertools.product(p, d, q))]

    if is_exog == True:

        min_aic = 999999999
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    model = sm.tsa.statespace.SARIMAX(df, exog=df_exog[df.index.min():df.index.max()],
                                                      order=param,
                                                      seasonal_order=param_seasonal,
                                                      enforce_stationarity=False,
                                                      enforce_invertibility=False,
                                                      freq='D')

                    results = model.fit()
                    print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))

                    # Check for best model with lowest AIC
                    if results.aic < min_aic:
                        min_aic = results.aic
                        min_aic_model = results
                except:
                    continue

        start_index = df.index.max() + dt.timedelta(1)
        end_index = start_index + dt.timedelta(days=forecast_steps - 1)

        predictions = pd.DataFrame(
            min_aic_model.predict(start=start_index, end=end_index, exog=df_exog[start_index:end_index]))
        predictions.columns = ['Counts']
        predictions[predictions < 0] = 0

        get_pred = min_aic_model.get_prediction(start=start_index, end=end_index, dynamic=False,
                                                exog=df_exog[start_index:end_index])
        predictions_ci = get_pred.conf_int(alpha=0.05)
        predictions_ci[predictions_ci < 0] = 0

    else:

        min_aic = 999999999
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    model = sm.tsa.statespace.SARIMAX(df,
                                                      order=param,
                                                      seasonal_order=param_seasonal,
                                                      enforce_stationarity=False,
                                                      enforce_invertibility=False,
                                                      freq='D')

                    results = model.fit()
                    print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))

                    # Check for best model with lowest AIC
                    if results.aic < min_aic:
                        min_aic = results.aic
                        min_aic_model = results
                except:
                    continue

        start_index = df.index.max() + dt.timedelta(1)
        end_index = start_index + dt.timedelta(days=forecast_steps - 1)

        predictions = pd.DataFrame(min_aic_model.predict(start=start_index, end=end_index))
        predictions.columns = ['Counts']
        predictions[predictions < 0] = 0

        get_pred = min_aic_model.get_prediction(start=start_index, end=end_index, dynamic=False)
        predictions_ci = get_pred.conf_int(alpha=0.05)
        predictions_ci[predictions_ci < 0] = 0

    return model, predictions, predictions_ci


def plot_model_diagnostics(model) -> True:
    """
    Plot model diagnostics from statsmodel model object
    :param model: model object from fit() function
    :return: plot of model diagnostics
    """
    model.plot_diagnostics(figsize=(15, 8))
    return True


def plot_forecast(df: pd.DataFrame, df_predictions: pd.DataFrame, df_conf_interval: pd.DataFrame):
    """
    Plots the given data and forecasts within 95% confidence interval. All dataframes should be date-indexed.
    :param df: given data
    :param df_predictions: forecasts/predictions
    :param df_conf_interval: date indexed dataframe with 2 columns, for lower and upper bounds of the predictions respectively
    """
    plt.figure(figsize=(15, 8))
    plt.plot(df, label='Data')
    plt.plot(df_predictions, label='Forecast')
    plt.fill_between(df_predictions.index,
                     df_conf_interval.iloc[:, 0],
                     df_conf_interval.iloc[:, 1], color='k', alpha=.2, label='Confidence Interval')
    plt.legend(loc='best')
    plt.title('Forecasts with 95% confidence for July 1st - 30th')
    plt.show()


def mean_absolute_percentage_error(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    """
    Calculate the mean absolute percentage error
    :param y_true: True data
    :param y_pred: Predictions
    :return: mape
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def evaluate_forecast(y: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate metrics such as mean absolute error, mean squared error,
    mean abs percentage error and root mean square error
    for evaluating the model
    :param y: true data
    :param pred: predictions from model
    :return:
    """
    results = pd.DataFrame({'r2_score': r2_score(y, pred),
                            }, index=[0])
    results['mean_absolute_error'] = mean_absolute_error(y, pred)
    results['mse'] = mean_squared_error(y, pred)
    results['mape'] = mean_absolute_percentage_error(y.Counts, pred.Counts)
    results['rmse'] = np.sqrt(results['mse'])
    return results


def main(visit_file_path: str, screening_file_path: str, screening_thresh: str) -> str:
    """
    main function that implements the required solution
    :param visit_file_path: str path to visits data
    :param screening_file_path: str path to screening data
    :param screening_thresh: screening threshold for alerting
    :return: True if number of covid screenings expected to go over the threshold
    """
    # read data
    df_visit, _ = read_data(visit_file_path)
    df_screening, _ = read_data(screening_file_path)

    # clean data
    df_visit['Date'] = pd.to_datetime(df_visit['Date'])
    df_visit.set_index('Date', inplace=True)

    df_screening['Date'] = pd.to_datetime(df_screening['Date'])
    df_screening.set_index('Date', inplace=True)
    df_screening_og = df_screening.copy()
    df_screening = df_screening[df_screening.Counts != 0]

    df_visit_train, df_visit_test = train_valid_test_split(df_visit)
    df_screening_train, df_screening_test = train_valid_test_split(df_screening)

    # evaluate model for visits
    model, visit_pred, visit_pred_ci = get_pred_from_sarimax(df_visit_train, False, pd.DataFrame(), len(df_visit_test))
    #     evaluate_forecast(df_visit_test, visit_pred)

    # get visit forecasts for july
    _, visit_forecasts, visit_forecasts_ci = get_pred_from_sarimax(df_visit, False, pd.DataFrame(), 31)
    #     plot_forecast(df_visit, visit_forecasts, visit_forecasts_ci)

    df_visits_forecasted = pd.concat([df_visit, visit_forecasts], axis=0)

    # evaluate model on screening
    model, screening_pred, screening_pred_ci = get_pred_from_sarimax(df_screening_train, False, pd.DataFrame(),
                                                                     len(df_screening_test))
    screening_model_metrics = evaluate_forecast(df_screening_test, screening_pred)

    # get screening forecasts for july using visits
    model, screening_forecasts, screening_forecasts_ci = get_pred_from_sarimax(df_screening, True, df_visits_forecasted,
                                                                               31)
    plot_forecast(df_screening_og, screening_forecasts, screening_forecasts_ci)

    if np.any(screening_forecasts.Counts) > screening_thresh:
        send_alert = True
    else:
        send_alert = False

    print("\n\n Send Alert : ", send_alert)
    print("\n\n Forecast evaluation metrics : \n\n", screening_model_metrics)

    return send_alert, screening_model_metrics

if __name__ == "__main__":
    main("visit_data.csv", "screening_data.csv", 300000)

