# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # #
# Finding the ARIMA model = EURO - 1 year #
# # # # # # # # # # # # # # # # # # # # # #

# In[0.1]: Importação dos pacotes

import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import seaborn as sns
import matplotlib.pyplot as plt


# In[0.2]: Import dataframes

df_eur_arima_1year = pd.read_csv("./datasets/arima_ready/eur_arima_1year.csv", float_precision="high", parse_dates=([0]))

df_eur_arima_1year.info()

# In[1]: Choosing the best ARIMA model

# Parameters:
# AR - Up to 2
# d = 1
# MA - Up to 2

eur_exog1 = pd.concat([df_eur_arima_1year['lag4'].shift(1), df_eur_arima_1year["lag6"].shift(1), df_eur_arima_1year["lag6"].shift(2)], axis=1).dropna()
eur_exog2 = pd.concat([df_eur_arima_1year["lag6"].shift(1), df_eur_arima_1year["lag6"].shift(2)], axis=1).dropna()
eur_exog3 = pd.concat([df_eur_arima_1year["lag4"].shift(1)], axis=1).dropna()
eur_exog4 = pd.concat([df_eur_arima_1year["lag6"].shift(1)], axis=1).dropna()

# After running tests with all exog combinations, it's clear that none of them add to the model.

# In[1.1]: Running ARIMA models by hand

# Trying AR(2), d = 1, MA(2)

eur_fit_1year_212 = ARIMA(df_eur_arima_1year['eur'], exog=None, order=(2,1,2), enforce_stationarity=True).fit()

eur_fit_1year_212.summary()


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
                               d=1,
                               test="adf",
                               m=1,
                               seasonal = False,
                               stepwise = True,
                               trace = True
                               ).fit(y = df_eur_arima_1year['eur'])

eur_fit_1year_AIC.summary()

"""
Performing stepwise search to minimize aic
 ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=-2117.102, Time=0.33 sec
 ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=-2115.981, Time=0.04 sec
 ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=-2115.326, Time=0.02 sec
 ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=-2115.738, Time=0.04 sec
 ARIMA(0,1,0)(0,0,0)[0]             : AIC=-2116.286, Time=0.02 sec
 ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=-2118.417, Time=0.22 sec
 ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=-2118.209, Time=0.13 sec
 ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=-2115.686, Time=0.09 sec
 ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=-2114.575, Time=0.30 sec
 ARIMA(0,1,3)(0,0,0)[0] intercept   : AIC=-2117.539, Time=0.15 sec
-> ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=-2118.535, Time=0.28 sec <-
 ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=-2116.978, Time=0.08 sec
 ARIMA(3,1,1)(0,0,0)[0] intercept   : AIC=-2114.551, Time=0.17 sec
 ARIMA(3,1,0)(0,0,0)[0] intercept   : AIC=-2116.417, Time=0.09 sec
 ARIMA(3,1,2)(0,0,0)[0] intercept   : AIC=-2112.778, Time=0.33 sec
 ARIMA(2,1,1)(0,0,0)[0]             : AIC=-2116.873, Time=0.11 sec

Best model:  ARIMA(2,1,1)(0,0,0)[0] intercept
Total fit time: 2.422 seconds
"""

eur_fit_1year_BIC = auto_arima(y = df_eur_arima_1year['eur'], 
                               d=1,
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
 ARIMA(2,1,2)(0,0,0)[0] intercept   : BIC=-2095.445, Time=0.32 sec
 ARIMA(0,1,0)(0,0,0)[0] intercept   : BIC=-2108.762, Time=0.04 sec
 ARIMA(1,1,0)(0,0,0)[0] intercept   : BIC=-2104.497, Time=0.02 sec
 ARIMA(0,1,1)(0,0,0)[0] intercept   : BIC=-2104.910, Time=0.03 sec
-> ARIMA(0,1,0)(0,0,0)[0]             : BIC=-2112.676, Time=0.02 sec <-
 ARIMA(1,1,1)(0,0,0)[0] intercept   : BIC=-2101.248, Time=0.07 sec

Best model:  ARIMA(0,1,0)(0,0,0)[0]          
Total fit time: 0.493 seconds
"""

eurdiff_fit_1year_AIC = auto_arima(y = df_eur_arima_1year['diff'],
                                   max_p = 2,
                                   max_q = 2,
                                   test = "adf",
                                   m = 1,
                                   seasonal = False,
                                   stepwise = True,
                                   trace = True
                                   ).fit(y = df_eur_arima_1year['diff'])

eurdiff_fit_1year_AIC.summary()

"""
Performing stepwise search to minimize aic
 ARIMA(2,0,2)(0,0,0)[0]             : AIC=-2122.115, Time=0.15 sec
 ARIMA(0,0,0)(0,0,0)[0]             : AIC=-2123.888, Time=0.02 sec
 ARIMA(1,0,0)(0,0,0)[0]             : AIC=-2122.923, Time=0.03 sec
 ARIMA(0,0,1)(0,0,0)[0]             : AIC=-2123.217, Time=0.03 sec
 ARIMA(1,0,1)(0,0,0)[0]             : AIC=-2124.583, Time=0.04 sec
 ARIMA(2,0,1)(0,0,0)[0]             : AIC=-2124.179, Time=0.08 sec
 ARIMA(1,0,2)(0,0,0)[0]             : AIC=-2123.947, Time=0.05 sec
-> ARIMA(0,0,2)(0,0,0)[0]             : AIC=-2125.016, Time=0.03 sec <-
 ARIMA(0,0,3)(0,0,0)[0]             : AIC=-2123.740, Time=0.06 sec
 ARIMA(1,0,3)(0,0,0)[0]             : AIC=-2122.047, Time=0.15 sec
-> ARIMA(0,0,2)(0,0,0)[0] intercept   : AIC=-2125.869, Time=0.08 sec <-
-> ARIMA(0,0,1)(0,0,0)[0] intercept   : AIC=-2123.395, Time=0.06 sec
 ARIMA(1,0,2)(0,0,0)[0] intercept   : AIC=-2125.664, Time=0.09 sec
 ARIMA(0,0,3)(0,0,0)[0] intercept   : AIC=-2125.062, Time=0.09 sec
 ARIMA(1,0,1)(0,0,0)[0] intercept   : AIC=-2120.839, Time=0.15 sec
 ARIMA(1,0,3)(0,0,0)[0] intercept   : AIC=-2123.817, Time=0.10 sec

Best model:  ARIMA(0,0,2)(0,0,0)[0] intercept
Total fit time: 1.217 seconds
"""

eurdiff_fit_1year_BIC = auto_arima(y = df_eur_arima_1year['diff'], 
                               information_criterion="bic",
                               test="adf",
                               m=1,
                               seasonal = False,
                               stepwise = True,
                               trace = True
                               ).fit(y = df_eur_arima_1year['diff'])

eurdiff_fit_1year_BIC.summary()

"""
Performing stepwise search to minimize bic
 ARIMA(2,0,2)(0,0,0)[0]             : BIC=-2104.050, Time=0.16 sec
-> ARIMA(0,0,0)(0,0,0)[0]             : BIC=-2120.275, Time=0.02 sec <-
 ARIMA(1,0,0)(0,0,0)[0]             : BIC=-2115.697, Time=0.03 sec
-> ARIMA(0,0,1)(0,0,0)[0]             : BIC=-2115.990, Time=0.03 sec <-
 ARIMA(1,0,1)(0,0,0)[0]             : BIC=-2113.743, Time=0.04 sec
 ARIMA(0,0,0)(0,0,0)[0] intercept   : BIC=-2116.523, Time=0.03 sec

Best model:  ARIMA(0,0,0)(0,0,0)[0]          
Total fit time: 0.305 seconds
"""

"""
Basing on AIC

|  AIC ARIMA(2,1,1) + int.   | AIC_diff ARIMA(0,0,2) + int. |
=============================================================
| Log Likelihood    1064.268 | Log Likelihood      1066.934 |
| AIC              -2118.535 | AIC                -2125.869 |
| BIC              -2100.488 | BIC                -2111.416 |
=============================================================

Basing on BIC

|     BIC ARIMA(0,1,0)      |   BIC_diff ARIMA(0,0,0)    |
==========================================================
| Log Likelihood   1059.143 | Log Likelihood    1062.944 |
| AIC             -2116.286 | AIC              -2123.888 |
| BIC             -2112.676 | BIC              -2120.275 |
==========================================================
"""

# Chosen Model 5 - ARIMA(0,0,2) + Intercept:
    
# Trying AR(0), d = 1, MA(2)

eur_fit_1year_002 = ARIMA(df_eur_arima_1year['diff'], exog=None, order=(0,0,2), enforce_stationarity=True).fit()

eur_fit_1year_002.summary()

"""
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                   diff   No. Observations:                  274
Model:                 ARIMA(0, 0, 2)   Log Likelihood                1066.934
Date:                Tue, 24 Sep 2024   AIC                          -2125.868
Time:                        01:31:21   BIC                          -2111.416
Sample:                             0   HQIC                         -2120.067
                                - 274                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const          0.0004      0.000      1.625      0.104   -8.08e-05       0.001
ma.L1         -0.0937      0.058     -1.610      0.107      -0.208       0.020
ma.L2         -0.1235      0.063     -1.975      0.048      -0.246      -0.001
sigma2      2.427e-05   1.48e-06     16.357      0.000    2.14e-05    2.72e-05
===================================================================================
Ljung-Box (L1) (Q):                   0.00   Jarque-Bera (JB):                62.43
Prob(Q):                              0.94   Prob(JB):                         0.00
Heteroskedasticity (H):               1.09   Skew:                             0.41
Prob(H) (two-sided):                  0.68   Kurtosis:                         5.19
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
"""
