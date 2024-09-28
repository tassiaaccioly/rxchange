# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # #
# Running the ARIMA model = EURO - 365 days #
# # # # # # # # # # # # # # # # # # # # # # #

# In[0.1]: Importação dos pacotes

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import norm, shapiro, jarque_bera
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import mean, stdev

# In[0.2]: Define important functions

# Function to invert the result value (inverse diff then inverse log)
def invert_results(series_diff, first_value):
    print(series_diff)
    print(first_value)
    series_inverted = np.r_[first_value, series_diff].cumsum().astype('float64')
    return np.exp(series_inverted)

# test_difffunc = pd.DataFrame({"eur": invert_results(df_eur_arima_train["diff"], eur_arima_1year.iloc[5,2])})

# In[0.2]: Import dataframes

eur_train_1year = pd.read_csv("./datasets/arima_ready/eur_train_1year.csv", float_precision="high", parse_dates=([0]))
 
eur_arima_train = eur_train_1year

eur_arima_train.info()

eur_test_1year = pd.read_csv("./datasets/arima_ready/eur_test_1year.csv", float_precision="high", parse_dates=([0]))
 
eur_arima_test = eur_test_1year

eur_arima_test.info()

y = eur_arima_train["diff"]

test_diff = df_eur_arima_test["log"]