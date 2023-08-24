from sklearn import metrics
import pandas as pd
import numpy as np


def compute_metrics(df : pd.DataFrame, # the df include two columns: (actual, forecast)
                    metrics_list: list # the list include every metrics you wanna calculate
                    ) -> dict:
    output = {}
    for m in metrics_list:
        output[m] = globals()[m](df)
    return output


def wbias(df):
    """
    weighted bias
    wbias = sum(Ai - Fi)/sum(Ai)
    """
    return np.nansum(df.actual - df.forecast)/np.nansum(df.actual)

def mape(df):
    """
    mean absolute error, using sklearn: https://scikit-learn.org/stable/modules/model_evaluation.html#mean-absolute-percentage-error
    mape = mean(|Ai - Fi|/|Ai|)
    """
    return metrics.mean_absolute_percentage_error(df.actual, df.forecast)

def wmape(df):
    """
    weighted mean absolute percentage error
    wmape = sum(|Ai - Fi|)/sum(|Ai|)
    """
    return np.nansum(np.abs(df.forecast - df.actual)) / np.nansum(np.abs(df.actual))

def rmse(df):
    """
    rooted mean squared error, using sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    rmse = sqrt(mean((Ai - Fi)^2))
    """
    return metrics.mean_squared_error(df.actual, df.forecast, squared = False)

def wuforec(df):
    """
    weighetd underforecasting: the magnitude of underforecasting normalized by absolute mean scale of the actuals
    """
    if sum(df.actual != 0) == 0:
        return np.nan

    under_df = df[df.forecast < df.actual]  # If forecast < actual then underforecast
    return (under_df.actual - under_df.forecast).sum() / np.abs(df.actual).sum()

def woforec(df):
    """
    weighetd overforecasting: the magnitude of underforecasting normalized by absolute mean scale of the actuals
    """
    if sum(df.actual != 0) == 0:
        return np.nan

    over_df = df[df.forecast > df.actual]  # If forecast > actual then underforecast
    return (over_df.forecast - over_df.actual).sum() / np.abs(df.actual).sum()