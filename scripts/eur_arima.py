# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # #
# Finding the ARIMA model = EURO - 1 year #
# # # # # # # # # # # # # # # # # # # # # #

# In[0.1]: Importação dos pacotes

import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import SARIMAX


# In[0.2]: Import dataframes

df_eur_arima_1year = pd.read_csv("./datasets/arima_ready/eur_arima_1year.csv", float_precision="high", parse_dates=([0]))

df_eur_arima_1year.info()

# In[1]: Choosing the best ARIMA model

# Parameters:
# AR - Up to 2
# d = 1
# MA - Up to 2

eur_exog1 = pd.concat([df_eur_arima_1year["lag6"].shift(1), df_eur_arima_1year["lag6"].shift(2)], axis=1).dropna()
eur_exog2 = pd.concat([df_eur_arima_1year["lag6"].shift(1)], axis=1).dropna()

len_eur_original = len(df_eur_arima_1year)
len_eur_exog1 = len_eur_original - len(eur_exog1) - 1
len_eur_exog2 = len_eur_original - len(eur_exog2) - 1

df_eur_exog1 = df_eur_arima_1year['eur'].drop(df_eur_arima_1year['eur'].loc[0:len_eur_exog1].index)
df_eur_exog2 = df_eur_arima_1year['eur'].drop(df_eur_arima_1year['eur'].loc[0:len_eur_exog2].index)

# After running tests with all exog combinations, it's clear that none of them add to the model.

# In[1.1]: Running ARIMA models by hand

# Trying AR(2), d = 1, MA(2)

eur_fit_1year_000 = SARIMAX(df_eur_arima_1year["diff"], exog=None, order=(0,0,0), enforce_stationarity=True).fit()

eur_fit_1year_000.summary()


# Trying AR(0), d = 1, MA(2)

eur_fit_1year_012 = ARIMA(df_eur_arima_1year['eur'], exog=None, order=(0,1,2), enforce_stationarity=True).fit()

eur_fit_1year_012.summary()


# Trying AR(0), d = 1, MA(1) // values suggested by previous research

eur_fit_1year_011 = ARIMA(df_eur_arima_1year['eur'], exog=None, order=(0,1,1), enforce_stationarity=True).fit()

eur_fit_1year_011.summary()

# Trying AR(2), d = 1, MA(2)

eur_fit_1year_202 = ARIMA(df_eur_arima_1year['diff'], exog=None, order=(2,0,2), enforce_stationarity=True).fit()

eur_fit_1year_202.summary()


# Trying AR(0), d = 1, MA(2)

eur_fit_1year_002 = ARIMA(df_eur_arima_1year['diff'], exog=None, order=(0,0,2), enforce_stationarity=True).fit()

eur_fit_1year_002.summary()


# Trying AR(0), d = 1, MA(1) // values suggested by previous research

eur_fit_1year_001 = ARIMA(df_eur_arima_1year['diff'], exog=None, order=(0,0,1), enforce_stationarity=True).fit()

eur_fit_1year_001.summary()

# Trying AR(0), d = 1, MA(0) // values suggested by previous research

eur_fit_1year_000 = ARIMA(df_eur_arima_1year['diff'], exog=None, order=(0,0,0), enforce_stationarity=True).fit()

eur_fit_1year_000.summary()


# In[1.2]: Comparing AIC, BIC and Loglike for the models:

"""
Differencing inside the ARIMA model: (d = 1)

|    Model 1 ARIMA(2,1,2)    |    Model 2 ARIMA(0,1,2)    |    Model 3 ARIMA(0,1,1)    |
========================================================================================
| Log Likelihood    1062.520 | Log Likelihood    1061.792 | Log Likelihood    1059.870 |
| AIC              -2115.040 | AIC              -2117.583 | AIC              -2115.739 |
| BIC              -2096.993 | BIC              -2106.755 | BIC              -2108.520 |
========================================================================================

Differencing before the ARIMA model: (d = 0)

|   Model 4 ARIMA(2,0,2)+i   |   Model 5 ARIMA(0,0,2)+i   |   Model 6 ARIMA(0,0,1)+i   |
========================================================================================
| Log Likelihood    1067.905 | Log Likelihood    1066.934 | Log Likelihood    1064.698 |
| AIC              -2123.811 | AIC              -2125.868 | AIC              -2123.396 |
| BIC              -2102.132 | BIC              -2111.416 | BIC              -2112.556 |
========================================================================================

Differencing before the ARIMA model: (d = 0)

|   Model 7 ARIMA(0,0,0)+i   |
==============================
| Log Likelihood    1063.874 |
| AIC              -2123.749 |
| BIC              -2116.522 |
==============================
"""

# In[1.3]: Confirming AIC and BIC for other (p,d,q) values with automated ARIMA

# we will be using the max parameters for p and q removed from the ACF and PACF
# plots: AR(2), MA(2)

eur_fit_1year_AIC = auto_arima(y = df_eur_arima_1year['eur'],
                               test="adf",
                               m=1,
                               seasonal = False,
                               stepwise = True,
                               trace = True
                               ).fit(y = df_eur_arima_1year['eur'])

eur_fit_1year_AIC.summary()

"""
Performing stepwise search to minimize aic
 ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=-1007.144, Time=0.27 sec
 ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=-1006.863, Time=0.03 sec
 ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=-1006.798, Time=0.03 sec
 ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=-1007.226, Time=0.05 sec
 ARIMA(0,1,0)(0,0,0)[0]             : AIC=-1006.879, Time=0.01 sec
-> ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=-1010.815, Time=0.21 sec <-
 ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=-1009.049, Time=0.24 sec
 ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=-1009.014, Time=0.28 sec
 ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=-1007.640, Time=0.16 sec
 ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=-1006.254, Time=0.03 sec
 ARIMA(1,1,1)(0,0,0)[0]             : AIC=-1008.312, Time=0.07 sec

Best model:  ARIMA(1,1,1)(0,0,0)[0] intercept
Total fit time: 1.372 seconds
"""

eur_fit_1year_BIC = auto_arima(y = df_eur_arima_1year['eur'],
                               information_criterion="bic",
                               test="adf",
                               m=1,
                               seasonal = False,
                               stepwise = True,
                               trace = True
                               ).fit(y = df_eur_arima_1year['eur'])

eur_fit_1year_BIC.summary()

"""
Performing stepwise search to minimize bic
 ARIMA(2,1,2)(0,0,0)[0] intercept   : BIC=-986.285, Time=0.26 sec
 ARIMA(0,1,0)(0,0,0)[0] intercept   : BIC=-999.910, Time=0.03 sec
 ARIMA(1,1,0)(0,0,0)[0] intercept   : BIC=-996.369, Time=0.03 sec
 ARIMA(0,1,1)(0,0,0)[0] intercept   : BIC=-996.797, Time=0.04 sec
-> ARIMA(0,1,0)(0,0,0)[0]             : BIC=-1003.403, Time=0.01 sec <-
 ARIMA(1,1,1)(0,0,0)[0] intercept   : BIC=-996.909, Time=0.16 sec

Best model:  ARIMA(0,1,0)(0,0,0)[0]          
Total fit time: 0.536 seconds
"""

eurdiff_fit_1year_AIC = auto_arima(y = df_eur_arima_1year['diff'].dropna(),
                                   test = "adf",
                                   m = 1,
                                   seasonal = False,
                                   stepwise = True,
                                   trace = True
                                   ).fit(y = df_eur_arima_1year['diff'].dropna())

eurdiff_fit_1year_AIC.summary()

"""
Performing stepwise search to minimize aic
 ARIMA(2,0,2)(0,0,0)[0]             : AIC=-1815.404, Time=0.10 sec
-> ARIMA(0,0,0)(0,0,0)[0]             : AIC=-1819.456, Time=0.02 sec <-
 ARIMA(1,0,0)(0,0,0)[0]             : AIC=-1818.958, Time=0.03 sec
 ARIMA(0,0,1)(0,0,0)[0]             : AIC=-1819.264, Time=0.03 sec
 ARIMA(1,0,1)(0,0,0)[0]             : AIC=-1815.456, Time=0.08 sec
 ARIMA(0,0,0)(0,0,0)[0] intercept   : AIC=-1819.367, Time=0.03 sec

Best model:  ARIMA(0,0,0)(0,0,0)[0]          
Total fit time: 0.291 seconds
"""

eurdiff_fit_1year_BIC = auto_arima(y = df_eur_arima_1year['diff'].dropna(), 
                               information_criterion="bic",
                               test="adf",
                               m=1,
                               seasonal = False,
                               stepwise = True,
                               trace = True
                               ).fit(y = df_eur_arima_1year['diff'].dropna())

eurdiff_fit_1year_BIC.summary()

"""
Performing stepwise search to minimize bic
 ARIMA(2,0,2)(0,0,0)[0]             : BIC=-1798.021, Time=0.10 sec
 ARIMA(0,0,0)(0,0,0)[0]             : BIC=-1815.979, Time=0.02 sec
 ARIMA(1,0,0)(0,0,0)[0]             : BIC=-1812.005, Time=0.03 sec
 ARIMA(0,0,1)(0,0,0)[0]             : BIC=-1812.311, Time=0.03 sec
 ARIMA(1,0,1)(0,0,0)[0]             : BIC=-1805.026, Time=0.07 sec
 ARIMA(0,0,0)(0,0,0)[0] intercept   : BIC=-1812.414, Time=0.03 sec

Best model:  ARIMA(0,0,0)(0,0,0)[0]          
Total fit time: 0.283 seconds
"""

"""
Basing on normal data order 1 differenced data in the auto_arima

|  AIC SARIMAX(1,1,1) + int. |    BIC SARIMAX(0,1,0)      |
===========================================================
| Log Likelihood     509.407 | Log Likelihood     504.440 |
| AIC              -1010.815 | AIC              -1006.879 |
| BIC               -996.909 | BIC              -1003.403 |
| intercept           0.0008 | intercept               NA |
| sigma2              0.0008 | sigma2              0.0009 |
===========================================================

Basing on order 0 differenced data

|  AIC_diff SARIMAX(0,0,0)  |  BIC_diff SARIMAX(0,0,0)   |
==========================================================
| Log Likelihood    910.728 | Log Likelihood     910.728 |
| AIC             -1819.456 | AIC              -1819.456 |
| BIC             -1815.979 | BIC              -1815.979 |
| sigma2          2.869e-05 | sigma2           2.869e-05 |
==========================================================
"""

# Chosen Model BIC_AIC SARIMAX(0,0,0)

eurdiff_fit_1year_BIC = auto_arima(y = df_eur_arima_1year['diff'].dropna(), 
                               information_criterion="bic",
                               test="adf",
                               m=1,
                               seasonal = False,2
                               stepwise = True,
                               trace = True
                               ).fit(y = df_eur_arima_1year['diff'].dropna())

eurdiff_fit_1year_BIC.summary()

"""
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                      y   No. Observations:                  239
Model:                        SARIMAX   Log Likelihood                 910.728
Date:                Thu, 26 Sep 2024   AIC                          -1819.456
Time:                        19:54:55   BIC                          -1815.979
Sample:                             0   HQIC                         -1818.055
                                - 239                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
sigma2      2.869e-05   1.96e-06     14.624      0.000    2.48e-05    3.25e-05
===================================================================================
Ljung-Box (L1) (Q):                   1.82   Jarque-Bera (JB):                25.42
Prob(Q):                              0.18   Prob(JB):                         0.00
Heteroskedasticity (H):               1.14   Skew:                             0.26
Prob(H) (two-sided):                  0.56   Kurtosis:                         4.51
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
"""

# Models were tested with exog, but all models with exog showed to have less BIC
# AIC and log-like than the ones without.