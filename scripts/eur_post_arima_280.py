# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # #
# Running the ARIMA model = EURO - 1 year #
# # # # # # # # # # # # # # # # # # # # # #

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

# Normal distribution:

nd = np.arange(-0.015, 0.020, 0.001)

mn = mean(nd)
sd = stdev(nd)


# In[0.2]: Import dataframes

eur_arima_1year = pd.read_csv("./datasets/arima_ready/eur_arima_1year.csv", float_precision="high", parse_dates=([0]))

df_eur_arima_train = eur_arima_1year

df_eur_arima_train.info()

df_eur_arima_test = pd.read_csv("./datasets/arima_ready/eur_arima_3months.csv", float_precision="high", parse_dates=([0]))

df_eur_arima_test.info()

y = df_eur_arima_train["diff"]

test_diff = df_eur_arima_test["log"]

# In[1.0]: Training the model SARIMAX(0,0,0)

eur_arima_final_000 = SARIMAX(y, exog=None, order=(0,0,0), seasonal_order=(0,0,0,0), enforce_stationarity=True, trend="c").fit()

eur_arima_final_000.summary()

"""
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                   diff   No. Observations:                  240
Model:                        SARIMAX   Log Likelihood                 910.728
Date:                Thu, 26 Sep 2024   AIC                          -1819.455
Time:                        20:09:37   BIC                          -1815.975
Sample:                             0   HQIC                         -1818.053
                                - 240                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
sigma2      2.863e-05   1.95e-06     14.652      0.000    2.48e-05    3.25e-05
===================================================================================
Ljung-Box (L1) (Q):                   1.82   Jarque-Bera (JB):                26.13
Prob(Q):                              0.18   Prob(JB):                         0.00
Heteroskedasticity (H):               1.14   Skew:                             0.26
Prob(H) (two-sided):                  0.56   Kurtosis:                         4.53
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
"""


eur_arima_final_000.plot_diagnostics(figsize=(10,7))
plt.show()



# In[1.1]: Saving and plotting the residuals for inspection

df_eur_arima_train["erros000"] = eur_arima_final_000.resid

erros000 = df_eur_arima_train["erros000"].dropna()

erros000.describe()

"""
count    239.000000
mean       0.000478
std        0.005346
min       -0.015484
25%       -0.002271
50%        0.000209
75%        0.002858
max        0.019281
Name: erros000, dtype: float64
"""

# Plot ACF and PACF of Residuals:

plt.figure(figsize=(11,8), dpi=300)
fig, (ax1, ax2) = plt.subplots(2)
plot_acf(erros000, ax=ax1)
plot_pacf(erros000, ax=ax2)

# Plot the scatter plot

plt.figure(figsize=(15, 10))
sns.scatterplot(x=df_eur_arima_train["date"], y=erros000, color="green", alpha=0.4,
             edgecolor=None)
plt.title("Resíduos do Modelo - ARIMA(0,0,2) + intercepto", fontsize="18")
plt.xlabel("")
plt.ylabel("")
plt.legend(fontsize=18, loc="upper right")
plt.show()


# In[1.2]: Run Residuals tests:

## Ljung-Box test

acorr_ljungbox(erros000, lags=[1, 2, 3, 4, 5, 6, 7, 14, 21, 30], return_df=True)

"""
      lb_stat  lb_pvalue
1    1.820730   0.177226
2    3.149544   0.207055
3    7.772643   0.050952
4    8.821635   0.065716
5    8.968990   0.110307
6   11.896027   0.064329
7   12.054932   0.098767
14  17.540155   0.228533
21  25.059296   0.244605
30  30.812866   0.424648
"""

# We can see that the p-value is higher than 0.05, so we DO NOT REJECT the
# null hypothesis, indicating there's no autocorrelation.
# it gets a little close to 0.05 but it not surpasses it on lag 3 and 4.

## SHAPIRO-WILK TEST

shapiro(erros000)

# Shapiro Result
#            statistic |                pvalue
#   0.9724904264599593 | 0.00013663174988855255

# p-value < 0.05 so we REJECT the null hypothesis, so we cannot confirm residuals
# are normally distribuited
# Shapiro-Wilk shows that the residuals are not normally distributed.

## JARQUE-BERA TEST

jarque_bera(erros000)

# Significance Result
#           statistic |                 pvalue
#   25.41505665000309 |  3.028241938954446e-06

# The p-value < 0.05 so we REJECT the null hypothesis and confirm residuals DO
# NOT FOLLOW A NORMAL DISTRIBUTION
# The Jarque-Bera test comproves the residuals are NOT NORMALLY DISTRIBUTED

# In[]: Predict

# testing the model

eur_arima_test_final = eur_arima_final_002int.apply(df_eur_arima_test["diff"])



# In[]: Testing the fitted values

df_eur_arima_train["yhat"] = eur_arima_final_002int.fittedvalues
eur_arima_fitted = pd.DataFrame(eur_arima_final_002int.fittedvalues)

# Fazendo previsões:

eur_arima_002_predict = eur_arima_final_002int.predict(0, 14 )

predicted = invert_results(eur_arima_002_predict, eur_arima_1year.iloc[5,2])


# In[]: 

# Plot the histogram

mu, std = norm.fit(erros000)


plt.figure(figsize=(15, 10))
sns.histplot(x=erros000, color="green", alpha=0.4,
             edgecolor=None, kde=True, line_kws={
                 "linewidth": 3, "linestyle": "dashed", "color": "m",
                 "label": "KDE"}, 
             )
xmin, xmax = plt.xlim()
x = np.linspace(-0.015, 0.020, len(erros000))
p = norm.pdf(x, mu, std)
plt.plot(x, p, "k", color="magenta", linewidth=3, label="Distribuição Normal")
plt.title("Resíduos do Modelo - ARIMA(0,0,2) + intercepto", fontsize="18")
plt.xlabel("")
plt.ylabel("")
plt.legend(fontsize=18, loc="upper right")
plt.show()

# Plot the scatter plot

plt.figure(figsize=(15, 10))
sns.scatterplot(x=df_eur_arima_train["date"], y=df_eur_arima_train["erros"], color="green", alpha=0.4,
             edgecolor=None)
plt.title("Resíduos do Modelo - ARIMA(0,0,2) + intercepto", fontsize="18")
plt.xlabel("")
plt.ylabel("")
plt.legend(fontsize=18, loc="upper right")
plt.show()