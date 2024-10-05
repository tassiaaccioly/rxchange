# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # #
# Running the ARIMA model = DOLLAR - 365 days #
# # # # # # # # # # # # # # # # # # # # # # # #

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

# test_difffunc = pd.DataFrame({"usd": invert_results(df_usd_arima_train["diff"], usd_arima_1year.iloc[5,2])})

# In[0.2]: Import dataframes

usd_train_1year = pd.read_csv("./datasets/arima_ready/usd_train_1year_365.csv", float_precision="high", parse_dates=([0]))
 
usd_arima_train = pd.DataFrame(usd_train_1year["usd"].values, index=usd_train_1year["date"], columns=["usd"])

usd_arima_train.info()

usd_test_1year = pd.read_csv("./datasets/arima_ready/usd_test_1year_365.csv", float_precision="high", parse_dates=([0]))
 
usd_arima_test = usd_test_1year

usd_arima_train_6 = pd.DataFrame(usd_train_1year["usd"].values, index=usd_train_1year["date"], columns=["usd"])[140:]

usd_arima_test.info()

y = usd_arima_train["usd"]

usd_exog1 = pd.DataFrame(pd.concat([usd_train_1year["eur"].shift(1)], axis=1).values, index=usd_train_1year["date"], columns=["eur"]).dropna()

len_usd_original = len(usd_arima_train)
len_usd_exog1 = len_usd_original - len(usd_exog1) - 1

df_usd_exog1 = usd_arima_train.drop(usd_arima_train.iloc[0:len_usd_exog1].index)

df_usd_exog = df_usd_exog1[1:]

# In[]: Run models

usd_365_010_fit = ARIMA(usd_arima_train_6, order=(0,1,0), enforce_stationarity=True).fit()

usd_365_010_fit.summary()

"""
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                    usd   No. Observations:                  298
Model:               SARIMAX(0, 1, 0)   Log Likelihood                 586.679
Date:                Sun, 29 Sep 2024   AIC                          -1171.358
Time:                        16:54:34   BIC                          -1167.664
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

usd_365_010_fit.plot_diagnostics(figsize=(10,7))
plt.show()

# In[]: Residuals

usd_arima_train_6["erros"] = usd_365_010_fit.resid

erros = usd_arima_train_6["erros"][1:]

sns.set_palette("viridis")
fig, ax = plt.subplots(1, figsize=(15,10), dpi=600)
mn = mean(erros)
strd = stdev(erros)
# Get parameters for the normal curve
x_pdf = np.linspace(-0.1, 0.1, 300)
y_pdf = norm.pdf(x_pdf, mn, strd)
ax.spines["left"].set(lw=3, color="black")
ax.spines["bottom"].set(lw=3, color="black")
sns.histplot(erros, stat="density", kde="true", label="Resíduos", line_kws={"label":"Est. de Densidade Kernel", "lw": 3})
ax.plot(x_pdf, y_pdf, lw=3, ls="--", label="Curva Normal")
plt.xticks(fontsize="22")
plt.yticks(fontsize="22")
plt.ylabel("Densidade", fontsize="22")
plt.xlabel("Erros", fontsize="22")
plt.legend(fontsize="22", loc="upper left")
plt.show()

erros.describe()

"""
count    297.000000
mean       0.002289
std        0.033543
min       -0.119050
25%       -0.014750
50%        0.000000
75%        0.016800
max        0.108800
Name: erros, dtype: float64
"""

# Plot ACF and PACF of Residuals:


fig, (ax1, ax2) = plt.subplots(2, figsize=(11,8), dpi=300)
plot_acf(erros, ax=ax1)
plot_pacf(erros, ax=ax2)

# Plot the scatter plot

plt.figure(figsize=(15, 10), dpi=300)
sns.scatterplot(x=usd_arima_train["date"], y=erros, color="green", alpha=0.4,
             edgecolor=None)
plt.title("Resíduos do Modelo - ARIMA(2,0,3) + intercepto", fontsize="18")
plt.xlabel("")
plt.ylabel("")
plt.legend(fontsize=18, loc="upper right")
plt.show()

# Plot an histogram

plt.figure(figsize=(15, 10), dpi=300)
sns.barplot(x=usd_arima_train["date"], y=erros, color="green", alpha=0.4,
             edgecolor=None)
plt.title("Resíduos do Modelo - SARIMAX(0,1,0)", fontsize="18")
plt.xlabel("")
plt.ylabel("")
plt.legend(fontsize=18, loc="upper right")
plt.show()

# In[]: Tests

acorr_ljungbox(erros, lags=[1, 2, 3, 4, 5, 6, 7, 14, 21, 30], return_df=True)

"""
      lb_stat  lb_pvalue
1    0.195174   0.658645
2    0.317746   0.853105
3    2.218907   0.528234
4    3.441013   0.486904
5    3.801241   0.578372
6    8.120884   0.229380
7    8.491259   0.291272
14  10.740788   0.706268
21  19.713701   0.539455
30  23.573174   0.790983
"""

# We can see that the p-value is higher than 0.05, so we DO NOT REJECT the
# null hypothesis, indicating there's no autocorrelation.

## SHAPIRO-WILK TEST

shapiro(erros)

# Shapiro Result
#            statistic |                pvalue
#   0.9652792111754172 |1.4762171183464121e-06

# p-value < 0.05 so we REJECT the null hypothesis, so we cannot confirm residuals
# are normally distribuited
# Shapiro-Wilk shows that the residuals are not normally distributed.

## JARQUE-BERA TEST

jarque_bera(erros)

# Significance Result
#           statistic |                 pvalue
#   27.73847074114976 |  9.476928637467217e-07

# The p-value < 0.05 so we REJECT the null hypothesis and confirm residuals DO
# NOT FOLLOW A NORMAL DISTRIBUTION
# The Jarque-Bera test comproves the residuals are NOT NORMALLY DISTRIBUTED