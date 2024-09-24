# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # #
# Running the ARIMA model = DOLLAR - 1 year #
# # # # # # # # # # # # # # # # # # # # # # #

# In[0.1]: Importação dos pacotes

import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import seaborn as sns
import matplotlib.pyplot as plt


# In[0.2]: Import dataframes

df_usd_arima_1year = pd.read_csv("./datasets/arima_ready/usd_arima_1year.csv", float_precision="high", parse_dates=([0]))

df_usd_arima_1year.info()

# In[1]: Choosing the best ARIMA model

# In[1.1]: Running ARIMA models by hand

usd_exog = pd.concat([df_usd_arima_1year['lag'].shift(1), df_usd_arima_1year['lag'].shift(2)], axis=1).dropna()

# Trying AR(2), d = 0, MA(2)

usd_fit_1year_202 = ARIMA(df_usd_arima_1year['usd'], exog=usd_exog, order=(2,0,2)()).fit()
 
usd_fit_1year_202.summary()
