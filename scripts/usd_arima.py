# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # #
# Finding the ARIMA model = DOLLAR - 1 year #
# # # # # # # # # # # # # # # # # # # # # # #

# In[0.1]: Importação dos pacotes

import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import SARIMAX
import seaborn as sns
import matplotlib.pyplot as plt


# In[0.2]: Import dataframes

usd_train_1year = pd.read_csv("./datasets/arima_ready/usd_train_1year_365.csv", float_precision="high", parse_dates=([0]))
 
usd_arima_train = usd_train_1year

usd_arima_train.info()

usd_test_1year = pd.read_csv("./datasets/arima_ready/usd_test_1year_365.csv", float_precision="high", parse_dates=([0]))
 
usd_arima_test = usd_test_1year

usd_arima_test.info()
# In[0.3] Define exogs:

# Normal dataset with normal exogs

usd_exog1 = pd.concat([usd_train_1year["diff"].shift(4)], axis=1).dropna()
usd_exog2 = pd.concat([usd_train_1year["diff"].shift(5)], axis=1).dropna()
usd_exog3 = pd.concat([usd_train_1year["diff"].shift(6)], axis=1).dropna()

len_usd_original = len(usd_train_1year)
len_usd_exog1 = len_usd_original - len(usd_exog1) - 1
len_usd_exog2 = len_usd_original - len(usd_exog2) - 1
len_usd_exog3 = len_usd_original - len(usd_exog3) - 1

df_usd_exog1 = usd_train_1year['usd'].drop(usd_train_1year['usd'].loc[0:len_usd_exog1].index)
df_usd_exog2 = usd_train_1year['usd'].drop(usd_train_1year['usd'].loc[0:len_usd_exog2].index)
df_usd_exog3 = usd_train_1year['usd'].drop(usd_train_1year['usd'].loc[0:len_usd_exog3].index)

# In[1.0]: Choosing the best ARIMA model

# PARAMETERS:
# AR = 2
# d = 0 ou 1
# MA = 2

# In[1.1]: Running automated arima functions:

usd_fit_1year_AIC = auto_arima(y = usd_train_1year['usd'], 
                               test="adf",
                               m=1,
                               seasonal = False,
                               stepwise = True,
                               trace = True
                               ).fit(y = usd_train_1year['usd'])

usd_fit_1year_AIC.summary()

"""
Performing stepwise search to minimize aic
 ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=-1165.035, Time=0.49 sec
 ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=-1170.742, Time=0.03 sec
 ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=-1168.935, Time=0.02 sec
 ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=-1168.944, Time=0.03 sec
-> ARIMA(0,1,0)(0,0,0)[0]             : AIC=-1171.358, Time=0.02 sec <-
 ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=-1166.939, Time=0.04 sec

Best model:  ARIMA(0,1,0)(0,0,0)[0]   
"""

usd_fit_1year_BIC = auto_arima(y = usd_train_1year['usd'], 
                               information_criterion="bic",
                               test="adf",
                               m=1,
                               seasonal = False,
                               stepwise = True,
                               trace = True
                               ).fit(y = usd_train_1year['usd'])

usd_fit_1year_BIC.summary()

"""
Performing stepwise search to minimize bic
 ARIMA(2,1,2)(0,0,0)[0] intercept   : BIC=-1142.872, Time=0.52 sec
 ARIMA(0,1,0)(0,0,0)[0] intercept   : BIC=-1163.355, Time=0.03 sec
 ARIMA(1,1,0)(0,0,0)[0] intercept   : BIC=-1157.854, Time=0.02 sec
 ARIMA(0,1,1)(0,0,0)[0] intercept   : BIC=-1157.863, Time=0.05 sec
-> ARIMA(0,1,0)(0,0,0)[0]             : BIC=-1167.664, Time=0.02 sec <-
 ARIMA(1,1,1)(0,0,0)[0] intercept   : BIC=-1152.164, Time=0.04 sec

Best model:  ARIMA(0,1,0)(0,0,0)[0]          
Total fit time: 0.679 seconds
"""

usddiff_fit_1year_AIC = auto_arima(y = usd_train_1year['diff'].dropna(),
                                   test = "adf",
                                   m = 1,
                                   seasonal = False,
                                   stepwise = True,
                                   trace = True
                                   ).fit(y = usd_train_1year['diff'].dropna())

usddiff_fit_1year_AIC.summary()

"""
Performing stepwise search to minimize aic
 ARIMA(2,0,2)(0,0,0)[0]             : AIC=-1163.571, Time=0.87 sec
-> ARIMA(0,0,0)(0,0,0)[0]             : AIC=-1171.358, Time=0.01 sec <-
 ARIMA(1,0,0)(0,0,0)[0]             : AIC=-1169.486, Time=0.02 sec
 ARIMA(0,0,1)(0,0,0)[0]             : AIC=-1169.490, Time=0.12 sec
 ARIMA(1,0,1)(0,0,0)[0]             : AIC=-1168.389, Time=0.09 sec
 ARIMA(0,0,0)(0,0,0)[0] intercept   : AIC=-1170.742, Time=0.03 sec

Best model:  ARIMA(0,0,0)(0,0,0)[0]          
Total fit time: 1.149 seconds
"""

usddiff_fit_1year_BIC = auto_arima(y = usd_train_1year['diff'].dropna(), 
                               information_criterion="bic",
                               test="adf",
                               m=1,
                               seasonal = False,
                               stepwise = True,
                               trace = True
                               ).fit(y = usd_train_1year['diff'].dropna())

usddiff_fit_1year_BIC.summary()

"""
Performing stepwise search to minimize bic
 ARIMA(2,0,2)(0,0,0)[0]             : BIC=-1145.103, Time=0.25 sec
 ARIMA(0,0,0)(0,0,0)[0]             : BIC=-1167.664, Time=0.01 sec
 ARIMA(1,0,0)(0,0,0)[0]             : BIC=-1162.098, Time=0.02 sec
 ARIMA(0,0,1)(0,0,0)[0]             : BIC=-1162.103, Time=0.01 sec
 ARIMA(1,0,1)(0,0,0)[0]             : BIC=-1157.307, Time=0.08 sec
 ARIMA(0,0,0)(0,0,0)[0] intercept   : BIC=-1163.355, Time=0.03 sec

Best model:  ARIMA(0,0,0)(0,0,0)[0]          
Total fit time: 0.403 seconds
"""


"""
No exogs

|    USD SARIMAX(0,1,0)      |   USD_diff SARIMAX(0,0,0)   |
============================================================
| Log Likelihood     586.679 | Log Likelihood      586.679 |
| AIC              -1171.358 | AIC               -1171.358 |
| BIC              -1167.664 | BIC               -1167.664 |
============================================================

"""

# In[]: Running automated arima functions with exogs:

usd_fit_1year_exog1 = auto_arima(y = df_usd_exog1, 
                               d=1,
                               test="adf",
                               m=1,
                               seasonal = False,
                               stepwise = True,
                               trace = True
                               ).fit(y = df_usd_exog1, exog=usd_exog1)

usd_fit_1year_exog1.summary()

"""
Performing stepwise search to minimize aic
 ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=inf, Time=2.02 sec
 ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=-1165.793, Time=0.03 sec
 ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=-1163.986, Time=0.02 sec
 ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=-1163.994, Time=0.04 sec
-> ARIMA(0,1,0)(0,0,0)[0]             : AIC=-1166.409, Time=0.01 sec <-
 ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=-1161.990, Time=0.07 sec

Best model:  ARIMA(0,1,0)(0,0,0)[0]          
Total fit time: 2.191 seconds
"""

usd_fit_1year_exog2 = auto_arima(y = df_usd_exog2, 
                               d=1,
                               test="adf",
                               m=1,
                               seasonal = False,
                               stepwise = True,
                               trace = True
                               ).fit(y = df_usd_exog2, exog=usd_exog2)

usd_fit_1year_exog2.summary()

"""
Performing stepwise search to minimize aic
-> ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=-1147.052, Time=2.27 sec <-
 ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=-1144.992, Time=0.02 sec
 ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=-1143.212, Time=0.04 sec
 ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=-1143.225, Time=0.03 sec
 ARIMA(0,1,0)(0,0,0)[0]             : AIC=-1145.963, Time=0.02 sec
 ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=-1139.283, Time=0.22 sec
 ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=-1139.418, Time=0.27 sec
 ARIMA(3,1,2)(0,0,0)[0] intercept   : AIC=-1138.839, Time=0.45 sec
 ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=-1144.533, Time=0.49 sec
 ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=-1141.218, Time=0.26 sec
 ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=-1139.925, Time=0.57 sec
 ARIMA(3,1,1)(0,0,0)[0] intercept   : AIC=-1139.618, Time=0.42 sec
 ARIMA(3,1,3)(0,0,0)[0] intercept   : AIC=-1145.433, Time=0.55 sec
 ARIMA(2,1,2)(0,0,0)[0]             : AIC=inf, Time=0.37 sec

Best model:  ARIMA(2,1,2)(0,0,0)[0] intercept
Total fit time: 5.982 seconds
"""

usd_fit_1year_exog3 = auto_arima(y = df_usd_exog3, 
                               d=1,
                               test="adf",
                               m=1,
                               seasonal = False,
                               stepwise = True,
                               trace = True
                               ).fit(y = df_usd_exog3, exog=usd_exog3)

usd_fit_1year_exog3.summary()

"""
Performing stepwise search to minimize aic
 ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=-1131.750, Time=0.59 sec
 ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=-1130.501, Time=0.02 sec
 ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=-1128.808, Time=0.02 sec
 ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=-1128.827, Time=0.04 sec
-> ARIMA(0,1,0)(0,0,0)[0]             : AIC=-1131.831, Time=0.01 sec <-
 ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=-1126.817, Time=0.11 sec

Best model:  ARIMA(0,1,0)(0,0,0)[0]          
Total fit time: 0.784 seconds
"""


"""
Differencing in the ARIMA model:  (d = 1) + EXOGS

|   Exog1 SARIMAX(0,1,2)     | Exog2 SARIMAX(2,1,2) + int  |    Exog3 SARIMAX(0,1,0)    |
========================================================================================
| Log Likelihood     584.205 | Log Likelihood      579.526 | Log Likelihood     566.915 |
| AIC              -1166.409 | AIC               -1147.052 | AIC              -1131.831 |
| BIC              -1162.719 | BIC               -1125.012 | BIC              -1128.175 |
========================================================================================
"""



# In[]: Comparing the best two models of each automated test:


"""
No exogs

|    USD SARIMAX(0,1,0)      |   USD_diff SARIMAX(0,0,0)   |
============================================================
| Log Likelihood     586.679 | Log Likelihood      586.679 |
| AIC              -1171.358 | AIC               -1171.358 |
| BIC              -1167.664 | BIC               -1167.664 |
============================================================

"""

"""
Differencing in the ARIMA model:  (d = 1) + EXOGS

|   Exog1 SARIMAX(0,1,2)     | Exog2 SARIMAX(2,1,2) + int  |    Exog3 SARIMAX(0,1,0)    |
========================================================================================
| Log Likelihood     584.205 | Log Likelihood      579.526 | Log Likelihood     566.915 |
| AIC              -1166.409 | AIC               -1147.052 | AIC              -1131.831 |
| BIC              -1162.719 | BIC               -1125.012 | BIC              -1128.175 |
========================================================================================
"""

# Chosen model: Model 1 ARIMA(0,1,0) (NO EXOGS)

usd_1year_fit = SARIMAX(endog=usd_train_1year['usd'],
                        exog=None,
                        seasonal_order=(0,0,0,0),
                        order=(0,1,0),
                        enforce_stationarity=True).fit()
 
usd_1year_fit.summary()

"""
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                    usd   No. Observations:                  298
Model:               SARIMAX(0, 1, 0)   Log Likelihood                 586.679
Date:                Sun, 29 Sep 2024   AIC                          -1171.358
Time:                        16:47:40   BIC                          -1167.664
Sample:                             0   HQIC                         -1169.879
                                - 298                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
sigma2         0.0011   6.99e-05     16.115      0.000       0.001       0.001
===================================================================================
Ljung-Box (L1) (Q):                   0.20   Jarque-Bera (JB):                27.74
Prob(Q):                              0.66   Prob(JB):                         0.00
Heteroskedasticity (H):               2.03   Skew:                             0.12
Prob(H) (two-sided):                  0.00   Kurtosis:                         4.48
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
"""
