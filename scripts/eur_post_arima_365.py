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

y1 = eur_arima_train["eur"]

eur_exog1 = pd.concat([eur_arima_train["lag7"].shift(1), eur_arima_train["lag7"].shift(3)], axis=1).dropna()

# In[]: Run models

eur_365_213_fit = SARIMAX(y1, exog=None, order=(2,1,3), seasonal_order=(0,0,0,0), enforce_stationarity=True, trend="c").fit()

eur_365_213_fit.summary()

"""
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                   diff   No. Observations:                  298
Model:               SARIMAX(2, 0, 3)   Log Likelihood                 593.844
Date:                Sat, 28 Sep 2024   AIC                          -1173.688
Time:                        10:02:46   BIC                          -1147.809
Sample:                             0   HQIC                         -1163.329
                                - 298                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
intercept      0.0014      0.001      1.806      0.071      -0.000       0.003
ar.L1          0.1329      0.212      0.627      0.531      -0.283       0.548
ar.L2          0.4238      0.237      1.789      0.074      -0.041       0.888
ma.L1         -0.1586      0.211     -0.752      0.452      -0.572       0.255
ma.L2         -0.4086      0.250     -1.637      0.102      -0.898       0.081
ma.L3         -0.1836      0.062     -2.980      0.003      -0.304      -0.063
sigma2         0.0011    7.2e-05     14.814      0.000       0.001       0.001
===================================================================================
Ljung-Box (L1) (Q):                   0.01   Jarque-Bera (JB):                23.60
Prob(Q):                              0.92   Prob(JB):                         0.00
Heteroskedasticity (H):               1.91   Skew:                             0.20
Prob(H) (two-sided):                  0.00   Kurtosis:                         4.32
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
"""

eur_365_213_fit.plot_diagnostics(figsize=(10,7))
plt.show()

# In[]: Residuals

eur_arima_train["erros"] = eur_365_213_fit.resid

erros = eur_arima_train["erros"][1:]

erros.describe()

"""
count    297.000000
mean       0.000168
std        0.032811
min       -0.098734
25%       -0.017140
50%       -0.001038
75%        0.015795
max        0.106948
Name: erros, dtype: float64
"""

# Plot ACF and PACF of Residuals:


fig, (ax1, ax2) = plt.subplots(2, figsize=(11,8), dpi=300)
plot_acf(erros, ax=ax1)
plot_pacf(erros, ax=ax2)

# Plot the scatter plot

plt.figure(figsize=(15, 10), dpi=300)
sns.scatterplot(x=eur_arima_train["date"], y=erros, color="green", alpha=0.4,
             edgecolor=None)
plt.title("Resíduos do Modelo - ARIMA(2,0,3) + intercepto", fontsize="18")
plt.xlabel("")
plt.ylabel("")
plt.legend(fontsize=18, loc="upper right")
plt.show()

# In[]: Tests

acorr_ljungbox(erros, lags=[1, 2, 3, 4, 5, 6, 7, 14, 21, 30], return_df=True)

"""
      lb_stat  lb_pvalue
1    0.010347   0.918981
2    0.011439   0.994297
3    0.017758   0.999374
4    0.136005   0.997790
5    0.692051   0.983402
6    2.305552   0.889563
7    2.380181   0.935846
14   4.423465   0.992332
21  13.923292   0.872872
30  19.065268   0.938631
"""

# We can see that the p-value is higher than 0.05, so we DO NOT REJECT the
# null hypothesis, indicating there's no autocorrelation.

## SHAPIRO-WILK TEST

shapiro(erros)

# Shapiro Result
#            statistic |                pvalue
#   0.9723413187939686 | 1.732172055425046e-05

# p-value < 0.05 so we REJECT the null hypothesis, so we cannot confirm residuals
# are normally distribuited
# Shapiro-Wilk shows that the residuals are not normally distributed.

## JARQUE-BERA TEST

jarque_bera(erros)

# Significance Result
#           statistic |                 pvalue
#   23.07077012674419 |  9.777907407038926e-06

# The p-value < 0.05 so we REJECT the null hypothesis and confirm residuals DO
# NOT FOLLOW A NORMAL DISTRIBUTION
# The Jarque-Bera test comproves the residuals are NOT NORMALLY DISTRIBUTED