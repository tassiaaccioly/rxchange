# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # #
# Finding the ARIMA model = DOLLAR - 1 year #
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

# In[0.3] Define exogs:

# Normal dataset with normal exogs

usd_exog1 = pd.concat([df_usd_arima_1year['lag'].shift(1), df_usd_arima_1year['lag'].shift(2), df_usd_arima_1year["eur"].shift(1)], axis=1).dropna()
usd_exog2 = pd.concat([df_usd_arima_1year['lag'].shift(1), df_usd_arima_1year['lag'].shift(2), df_usd_arima_1year['lag'].shift(2), df_usd_arima_1year["eur"].shift(7)], axis=1).dropna()
usd_exog3 = pd.concat([df_usd_arima_1year['eur'].shift(15), df_usd_arima_1year['eur'].shift(16)], axis=1).dropna()

len_usd_original = len(df_usd_arima_1year)
len_usd_exog1 = len_usd_original - len(usd_exog1) - 1
len_usd_exog2 = len_usd_original - len(usd_exog2) - 1
len_usd_exog3 = len_usd_original - len(usd_exog3) - 1

df_usd_exog1 = df_usd_arima_1year['usd'].drop(df_usd_arima_1year['usd'].loc[0:len_usd_exog1].index)
df_usd_exog2 = df_usd_arima_1year['usd'].drop(df_usd_arima_1year['usd'].loc[0:len_usd_exog2].index)
df_usd_exog3 = df_usd_arima_1year['usd'].drop(df_usd_arima_1year['usd'].loc[0:len_usd_exog3].index)

# Differenced dataset with differenced exogs

df_usd_arima_1year["diffLag"] = df_usd_arima_1year["lag"].diff()

usddiff_exog1 = pd.concat([df_usd_arima_1year["diffLag"].shift(1), df_usd_arima_1year["diffLag"].shift(2), df_usd_arima_1year["eurDiff"].shift(1), df_usd_arima_1year["eurDiff"].shift(2)], axis=1).dropna()
usddiff_exog2 = pd.concat([df_usd_arima_1year["diffLag"].shift(1), df_usd_arima_1year["diffLag"].shift(2), df_usd_arima_1year["eurDiff"].shift(1)], axis=1).dropna()
usddiff_exog3 = pd.concat([df_usd_arima_1year["eurDiff"].shift(1), df_usd_arima_1year["eurDiff"].shift(2)], axis=1).dropna()
usddiff_exog4 = pd.concat([df_usd_arima_1year["eurDiff"].shift(1)], axis=1).dropna()
    
len_usd_original = len(df_usd_arima_1year)
len_usddiff_exog1 = len_usd_original - len(usddiff_exog1) - 1
len_usddiff_exog2 = len_usd_original - len(usddiff_exog2) - 1
len_usddiff_exog3 = len_usd_original - len(usddiff_exog3) - 1
len_usddiff_exog4 = len_usd_original - len(usddiff_exog4) - 1

df_usddiff_exog1 = df_usd_arima_1year["diff"].drop(df_usd_arima_1year["diff"].loc[0:len_usddiff_exog1].index)
df_usddiff_exog2 = df_usd_arima_1year["diff"].drop(df_usd_arima_1year["diff"].loc[0:len_usddiff_exog2].index)
df_usddiff_exog3 = df_usd_arima_1year["diff"].drop(df_usd_arima_1year["diff"].loc[0:len_usddiff_exog3].index)
df_usddiff_exog4 = df_usd_arima_1year["diff"].drop(df_usd_arima_1year["diff"].loc[0:len_usddiff_exog4].index)

# In[1]: Choosing the best ARIMA model

# PARAMETERS:
# AR = 2
# d = 0 ou 1
# MA = 2

# In[1.1]: Running ARIMA models by hand, without exogs

# Trying AR(2), d = 0, MA(2)

usd_fit_1year_012 = ARIMA(df_usd_arima_1year['usd'], exog=None, order=(0,1,2), enforce_stationarity=True).fit()
 
usd_fit_1year_012.summary()

usd_fit_1year_010 = ARIMA(df_usd_arima_1year['usd'], exog=None, order=(0,1,0), enforce_stationarity=True).fit()
 
usd_fit_1year_010.summary()

usd_fit_1year_002 = ARIMA(df_usd_arima_1year['diff'], exog=None, order=(0,0,2), enforce_stationarity=True).fit()
 
usd_fit_1year_002.summary()

usd_fit_1year_000 = ARIMA(df_usd_arima_1year['diff'], exog=None, order=(0,0,0), enforce_stationarity=True).fit()
 
usd_fit_1year_000.summary()

# Comparing AIC, BIC and Loglike and sigma2 for the models:

"""
Differencing inside the ARIMA model: (d = 1)

|    Model 1 ARIMA(0,1,2)    |    Model 2 ARIMA(0,1,0)    |
===========================================================
| Log Likelihood    1042.672 | Log Likelihood    1038.221 |
| AIC              -2079.344 | AIC              -2074.443 |
| BIC              -2068.516 | BIC              -2070.833 |
| sigma2           2.818e-05 | sigma2           2.907e-05 |
===========================================================

Differencing before the ARIMA model: (d = 0)

|   Model 3 ARIMA(0,0,2)+i   |   Model 4 ARIMA(0,0,0)+i   |
===========================================================
| Log Likelihood    1045.705 | Log Likelihood    1040.970 |
| AIC              -2083.410 | AIC              -2077.941 |
| BIC              -2068.958 | BIC              -2070.715 |
| intercept           0.0003 | intercept           0.0004 |
| sigma2           2.835e-05 | sigma2            2.93e-05 |
===========================================================
"""
# In[1.2]: Running ARIMA models by hand with exogs

usd_fit_1year_002_exog1 = ARIMA(df_usddiff_exog1, exog=usddiff_exog1, order=(0,0,2), enforce_stationarity=True, trend="c").fit()
 
usd_fit_1year_002_exog1.summary()

usd_fit_1year_002_exog2 = ARIMA(df_usddiff_exog2, exog=usddiff_exog2, order=(0,0,2), enforce_stationarity=True, trend="c").fit()
 
usd_fit_1year_002_exog2.summary()

usd_fit_1year_002_exog3 = ARIMA(df_usddiff_exog3, exog=usddiff_exog3, order=(0,0,2), enforce_stationarity=True, trend="c").fit()

usd_fit_1year_002_exog3.summary()

usd_fit_1year_002_exog4 = ARIMA(df_usddiff_exog4, exog=usddiff_exog4, order=(0,0,2), enforce_stationarity=True, trend="c").fit()

usd_fit_1year_002_exog4.summary()

usd_fit_1year_000_exog1 = ARIMA(df_usddiff_exog1, exog=usddiff_exog1, order=(0,0,0), enforce_stationarity=True, trend="c").fit()

usd_fit_1year_000_exog1.summary()

usd_fit_1year_000_exog2 = ARIMA(df_usddiff_exog2, exog=usddiff_exog2, order=(0,0,0), enforce_stationarity=True, trend="c").fit()

usd_fit_1year_000_exog2.summary()

usd_fit_1year_000_exog3 = ARIMA(df_usddiff_exog3, exog=usddiff_exog3, order=(0,0,0), enforce_stationarity=True, trend="c").fit()

usd_fit_1year_000_exog3.summary()

usd_fit_1year_000_exog4 = ARIMA(df_usddiff_exog4, exog=usddiff_exog4, order=(0,0,0), enforce_stationarity=True, trend="c").fit()

usd_fit_1year_000_exog4.summary()

# Comparing the AIC, BIC, Loglike and sigma2 for the models

"""
Differencing before the ARIMA model: (0,0,2) + intercept

|     Exog1 ARIMA(0,0,2)     |     Exog2 ARIMA(0,0,2)     |     Exog3 ARIMA(0,0,2)     |
========================================================================================
| Log Likelihood    1039.441 | Log Likelihood    1039.140 | Log Likelihood    1039.244 |
| AIC              -2062.881 | AIC              -2064.280 | AIC              -2066.488 |
| BIC              -2034.064 | BIC              -2039.066 | BIC              -2044.853 |
| intercept           0.0005 | intercept           0.0004 | intercept           0.0004 |
| sigma2           2.729e-05 | sigma2           2.735e-05 | sigma2           2.809e-05 |
========================================================================================

Differencing before the ARIMA model: (0,0,0) + intercept

|     Exog1 ARIMA(0,0,0)     |     Exog2 ARIMA(0,0,0)     |     Exog3 ARIMA(0,0,0)     |
========================================================================================
| Log Likelihood    1038.981 | Log Likelihood    1035.542 | Log Likelihood    1038.608 |
| AIC              -2065.962 | AIC              -2061.084 | AIC              -2069.216 |
| BIC              -2044.349 | BIC              -2043.073 | BIC              -2054.793 |
| intercept           0.0005 | intercept           0.0004 | intercept           0.0004 |
| sigma2           2.743e-05 | sigma2           2.813e-05 | sigma2           2.828e-05 |
========================================================================================

Differencing before the ARIMA model: (d = 0)

|    Exog4 ARIMA(0,0,2)+i    |    Exog4 ARIMA(0,0,0)+i    |
===========================================================
| Log Likelihood    1043.860 | Log Likelihood    1039.130 |
| AIC              -2077.720 | AIC              -2072.261 |
| BIC              -2059.673 | BIC              -2061.433 |
| intercept           0.0003 | intercept           0.0005 |
| sigma2           2.791e-05 | sigma2           2.893e-05 |
===========================================================

"""

# In[]: Running automated arima functions:

usd_fit_1year_AIC = auto_arima(y = df_usd_arima_1year['usd'], 
                               d=1,
                               test="adf",
                               m=1,
                               seasonal = False,
                               stepwise = True,
                               trace = True
                               ).fit(y = df_usd_arima_1year['usd'])

usd_fit_1year_AIC.summary()


usd_fit_1year_BIC = auto_arima(y = df_usd_arima_1year['usd'], 
                               d=1,
                               information_criterion="bic",
                               test="adf",
                               m=1,
                               seasonal = False,
                               stepwise = True,
                               trace = True
                               ).fit(y = df_usd_arima_1year['usd'])

usd_fit_1year_BIC.summary()

usddiff_fit_1year_AIC = auto_arima(y = df_usd_arima_1year['diff'],
                                   test = "adf",
                                   m = 1,
                                   seasonal = False,
                                   stepwise = True,
                                   trace = True
                                   ).fit(y = df_usd_arima_1year['diff'])

usddiff_fit_1year_AIC.summary()


usddiff_fit_1year_BIC = auto_arima(y = df_usd_arima_1year['diff'], 
                               information_criterion="bic",
                               test="adf",
                               m=1,
                               seasonal = False,
                               stepwise = True,
                               trace = True
                               ).fit(y = df_usd_arima_1year['diff'])

usddiff_fit_1year_BIC.summary()


"""
Basing on AIC (no exogs)

|      AIC ARIMA(0,1,2)      |    AIC_diff ARIMA(0,0,2)    |
============================================================
| Log Likelihood    1042.672 | Log Likelihood     1044.726 |
| AIC              -2079.344 | AIC               -2083.452 |
| BIC              -2068.516 | BIC               -2072.612 |
============================================================

Basing on BIC

|     BIC ARIMA(0,1,0)      |   BIC_diff ARIMA(0,0,0)    |
==========================================================
| Log Likelihood   1038.221 | Log Likelihood    1040.322 |
| AIC             -2074.443 | AIC              -2078.644 |
| BIC             -2070.833 | BIC              -2075.031 |
==========================================================
"""

# In[]: Running automated arima functions with exogs:

usd_fit_1year_BIC_exog1 = auto_arima(y = df_usd_exog1, 
                               d=1,
                               test="adf",
                               m=1,
                               seasonal = False,
                               stepwise = True,
                               trace = True
                               ).fit(y = df_usd_exog1, exog=usd_exog1)

usd_fit_1year_BIC_exog1.summary()

usd_fit_1year_BIC_exog2 = auto_arima(y = df_usd_exog2, 
                               d=1,
                               test="adf",
                               m=1,
                               seasonal = False,
                               stepwise = True,
                               trace = True
                               ).fit(y = df_usd_exog2, exog=usd_exog2)

usd_fit_1year_BIC_exog2.summary()


usd_fit_1year_BIC_exog3 = auto_arima(y = df_usd_exog3, 
                               d=1,
                               test="adf",
                               m=1,
                               seasonal = False,
                               stepwise = True,
                               trace = True
                               ).fit(y = df_usd_exog3, exog=usd_exog3)

usd_fit_1year_BIC_exog3.summary()


"""
Differencing before the ARIMA model: (d = 1) + EXOGS (AIC)

|   Exog1A SARIMAX(0,1,2)    |   Exog2A SARIMAX(0,0,2)    |   Exog3A SARIMAX(0,1,2)    |
========================================================================================
| Log Likelihood    1034.049 | Log Likelihood    1021.324 | Log Likelihood     988.978 |
| AIC              -2062.099 | AIC              -2036.648 | AIC              -1971.956 |
| BIC              -2051.292 | BIC              -2025.897 | BIC              -1961.309 |
========================================================================================

Differencing before the ARIMA model: (d = 1) + EXOGS (BIC)

|   Exog1B SARIMAX(0,1,0)    |   Exog2B SARIMAX(0,1,0)    |   Exog3B SARIMAX(0,1,0)    |
========================================================================================
| Log Likelihood    1029.681 | Log Likelihood    1016.087 | Log Likelihood     983.303 |
| AIC              -2057.362 | AIC              -2030.173 | AIC              -1964.607 |
| BIC              -2053.760 | BIC              -2026.590 | BIC              -1961.058 |
========================================================================================
"""

# In[]: Running automated arima functions with diff exogs:

# We're only doing already differenced here because in the previous tests, the
# difference values got the best scores.

usd_fit_1year_BIC_exog1 = auto_arima(y = df_usddiff_exog1, 
                               d=0,
                               test="adf",
                               m=1,
                               information_criterion="bic",
                               seasonal = False,
                               stepwise = True,
                               trace = True
                               ).fit(y = df_usddiff_exog1, exog=usddiff_exog1)

usd_fit_1year_BIC_exog1.summary()

usd_fit_1year_BIC_exog2 = auto_arima(y = df_usddiff_exog2, 
                               d=0,
                               test="adf",
                               m=1,
                               information_criterion="bic",
                               seasonal = False,
                               stepwise = True,
                               trace = True
                               ).fit(y = df_usddiff_exog2, exog=usddiff_exog2)

usd_fit_1year_BIC_exog2.summary()


usd_fit_1year_BIC_exog3 = auto_arima(y = df_usddiff_exog3, 
                               d=0,
                               test="adf",
                               m=1,
                               information_criterion="bic",
                               seasonal = False,
                               stepwise = True,
                               trace = True
                               ).fit(y = df_usddiff_exog3, exog=usddiff_exog3)

usd_fit_1year_BIC_exog3.summary()

usd_fit_1year_BIC_exog4 = auto_arima(y = df_usddiff_exog4, 
                               d=0,
                               test="adf",
                               m=1,
                               information_criterion="bic",
                               seasonal = False,
                               stepwise = True,
                               trace = True
                               ).fit(y = df_usddiff_exog4, exog=usddiff_exog4)

usd_fit_1year_BIC_exog4.summary()


"""
Differencing before the ARIMA model: (d = 0) + DIFFEXOGS (AIC)

|   Exog1A SARIMAX(0,0,2)    |   Exog2A SARIMAX(0,0,2)    |   Exog3A SARIMAX(0,0,2)    |
========================================================================================
| Log Likelihood    1038.399 | Log Likelihood    1025.481 | Log Likelihood     992.628 |
| AIC              -2070.799 | AIC              -2044.962 | AIC              -1979.257 |
| BIC              -2059.981 | BIC              -2034.200 | BIC              -1968.598 |
========================================================================================

Differencing before the ARIMA model: (d = 0) + DIFFEXOGS (BIC)

|   Exog1B SARIMAX(0,0,0)    |   Exog2B SARIMAX(0,0,0)    |   Exog3B SARIMAX(0,0,0)    |
========================================================================================
| Log Likelihood    1033.982 | Log Likelihood    1020.308 | Log Likelihood     986.947 |
| AIC              -2065.963 | AIC              -2038.616 | AIC              -1971.894 |
| BIC              -2062.357 | BIC              -2035.029 | BIC              -1968.341 |
========================================================================================
"""

# In[]: Comparing the best two models of each automated test:


"""
Differencing before the arima Model: (d = 0)

|   Model 1 SARIMAX(0,0,0)   |  Model 2 SARIMAX(0,0,2)   |
==========================================================
| Log Likelihood    1040.322 | Log Likelihood   1044.726 |
| AIC              -2078.644 | AIC             -2083.452 |
| BIC              -2075.031 | BIC             -2072.612 |
| sigma2           2.949e-05 | sigma2            2.93e-05 |
==========================================================

Differencing in the arima Model: (d = 1)

|   Model 3 SARIMAX(0,1,0)  |   Model 4 SARIMAX(0,1,2)   |
==========================================================
| Log Likelihood   1038.221 | Log Likelihood    1042.672 |
| AIC             -2074.443 | AIC              -2079.344 |
| BIC             -2070.833 | BIC              -2068.516 |
==========================================================
"""

"""
Differencing before the ARIMA model: (d = 0) + DIFFEXOGS

|   Exog1A SARIMAX(0,0,2)    |   Exog1B SARIMAX(0,0,0)    |
===========================================================
| Log Likelihood    1038.399 | Log Likelihood    1033.982 |
| AIC              -2070.799 | AIC              -2065.963 |
| BIC              -2059.981 | BIC              -2062.357 |
===========================================================

Differencing in the ARIMA model: (d = 1) + EXOGS

|   Exog1A SARIMAX(0,1,2)    |   Exog1B SARIMAX(0,1,0)    |
===========================================================
| Log Likelihood    1034.049 | Log Likelihood    1029.681 |
| AIC              -2062.099 | AIC              -2057.362 |
| BIC              -2051.292 | BIC              -2053.760 |
===========================================================
"""

"""
|   Model 3 ARIMA(0,0,2)+i   |   Model 4 ARIMA(0,0,0)+i   |
===========================================================
| Log Likelihood    1045.705 | Log Likelihood    1040.970 |
| AIC              -2083.410 | AIC              -2077.941 |
| BIC              -2068.958 | BIC              -2070.715 |
| intercept           0.0003 | intercept           0.0004 |
| sigma2           2.835e-05 | sigma2            2.93e-05 |
===========================================================

Handmande Differencing before the ARIMA model: (d = 0) + Exogs

|    Exog4 ARIMA(0,0,2)+i    |    Exog4 ARIMA(0,0,0)+i    |
===========================================================
| Log Likelihood    1043.860 | Log Likelihood    1039.130 |
| AIC              -2077.720 | AIC              -2072.261 |
| BIC              -2059.673 | BIC              -2061.433 |
| intercept           0.0003 | intercept           0.0005 |
| sigma2           2.791e-05 | sigma2           2.893e-05 |
===========================================================
"""

# Chosen model: Model 3 ARIMA(0,0,2) + intercept (NO EXOGS)

usd_fit_1year_002 = ARIMA(df_usd_arima_1year['diff'], exog=None, order=(0,0,2), enforce_stationarity=True).fit()
 
usd_fit_1year_002.summary()

"""
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                   diff   No. Observations:                  274
Model:                 ARIMA(0, 0, 2)   Log Likelihood                1045.705
Date:                Tue, 24 Sep 2024   AIC                          -2083.410
Time:                        20:33:31   BIC                          -2068.958
Sample:                             0   HQIC                         -2077.610
                                - 274                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const          0.0003      0.000      1.360      0.174      -0.000       0.001
ma.L1         -0.0472      0.060     -0.785      0.433      -0.165       0.071
ma.L2         -0.1824      0.062     -2.939      0.003      -0.304      -0.061
sigma2      2.835e-05   1.83e-06     15.509      0.000    2.48e-05    3.19e-05
===================================================================================
Ljung-Box (L1) (Q):                   0.00   Jarque-Bera (JB):                48.88
Prob(Q):                              0.94   Prob(JB):                         0.00
Heteroskedasticity (H):               1.10   Skew:                             0.42
Prob(H) (two-sided):                  0.66   Kurtosis:                         4.89
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
"""
