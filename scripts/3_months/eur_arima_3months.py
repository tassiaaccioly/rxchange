# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # #
# Finding the ARIMA model = EURO - 3 months #
# # # # # # # # # # # # # # # # # # # # # # #

# In[0.1]: Importação dos pacotes

import numpy as np
import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.api import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import shapiro, jarque_bera
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# In[0.2]: Import dataframes

df_eur_arima_3months = pd.read_csv("./datasets/arima_ready/eur_train_3months.csv", float_precision="high", parse_dates=([0]))

df_eur_arima_3months.info()

eur_train_arima = pd.DataFrame(df_eur_arima_3months.drop(["date"], axis=1).values, index=pd.date_range(start="2024-06-24", periods=45, freq="B"), columns=["eur", "log", "diff", "lag2"])

y = eur_train_arima["log"]

def invert_series(series):
    return np.exp(series)

# In[1.0]: Chosing and training the arima model

arima_3months_test = auto_arima(y,
                                test="adf",
                                m=1,
                                seasonal = False,
                                stepwise = True,
                                trace = True).fit(y)

arima_3months_test.summary()

"""
Performing stepwise search to minimize aic
 ARIMA(2,2,2)(0,0,0)[0] intercept   : AIC=-118.935, Time=0.10 sec
 ARIMA(0,2,0)(0,0,0)[0] intercept   : AIC=-109.618, Time=0.01 sec
 ARIMA(1,2,0)(0,0,0)[0] intercept   : AIC=-116.662, Time=0.02 sec
 ARIMA(0,2,1)(0,0,0)[0] intercept   : AIC=inf, Time=0.05 sec
 ARIMA(0,2,0)(0,0,0)[0]             : AIC=-111.540, Time=0.01 sec
 ARIMA(1,2,2)(0,0,0)[0] intercept   : AIC=inf, Time=0.11 sec
 ARIMA(2,2,1)(0,0,0)[0] intercept   : AIC=inf, Time=0.09 sec
 ARIMA(3,2,2)(0,0,0)[0] intercept   : AIC=-117.325, Time=0.12 sec
 ARIMA(2,2,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.13 sec
 ARIMA(1,2,1)(0,0,0)[0] intercept   : AIC=inf, Time=0.08 sec
 ARIMA(1,2,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.12 sec
 ARIMA(3,2,1)(0,0,0)[0] intercept   : AIC=inf, Time=0.10 sec
 ARIMA(3,2,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.13 sec
 ARIMA(2,2,2)(0,0,0)[0]             : AIC=inf, Time=0.17 sec

Best model:  ARIMA(2,2,2)(0,0,0)[0] intercept
Total fit time: 1.255 seconds
"""

arima_3months_test = SARIMAX(y, exog=None, order=(2,1,2), seasonal_order=(0,0,0,0), enforce_stationarity=True, trend="c").fit()

arima_3months_test.summary()

"""
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                    log   No. Observations:                   45
Model:               SARIMAX(2, 1, 2)   Log Likelihood                 150.800
Date:                Sun, 29 Sep 2024   AIC                           -289.599
Time:                        23:59:40   BIC                           -278.894
Sample:                    06-24-2024   HQIC                          -285.629
                         - 08-23-2024                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
intercept      0.0025      0.004      0.649      0.517      -0.005       0.010
ar.L1         -0.4331      0.463     -0.936      0.349      -1.340       0.474
ar.L2         -0.3192      0.343     -0.932      0.351      -0.991       0.352
ma.L1          0.7308      0.355      2.059      0.040       0.035       1.427
ma.L2          0.7551      0.287      2.635      0.008       0.194       1.317
sigma2      6.034e-05   1.45e-05      4.175      0.000     3.2e-05    8.87e-05
===================================================================================
Ljung-Box (L1) (Q):                   0.06   Jarque-Bera (JB):                 0.59
Prob(Q):                              0.81   Prob(JB):                         0.74
Heteroskedasticity (H):               0.82   Skew:                             0.28
Prob(H) (two-sided):                  0.71   Kurtosis:                         2.98
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
"""

arima_3months_test.plot_diagnostics(figsize=(10,7))
plt.show()

# In[1.1]: Saving and plotting the residuals for inspection

eur_train_arima["erros"] = arima_3months_test.resid

eur_train_arima["fitted"] = arima_3months_test.fittedvalues

erros = eur_train_arima["erros"][1:]

erros.describe()

"""
count    44.000000
mean      0.000092
std       0.007967
min      -0.015234
25%      -0.004854
50%      -0.000598
75%       0.005871
max       0.019507
Name: erros, dtype: float64
"""

# Plot ACF and PACF of Residuals:

fig, (ax1, ax2) = plt.subplots(2, figsize=(11,8), dpi=300)
plot_acf(erros, ax=ax1)
plot_pacf(erros, ax=ax2)

# Plot the scatter plot

plt.figure(figsize=(15, 10))
sns.scatterplot(erros, color="green", alpha=0.4,
             edgecolor=None)
plt.title("Resíduos do Modelo - ARIMA(2,1,2) + intercepto", fontsize="18")
plt.xlabel("")
plt.ylabel("")
plt.legend(fontsize=18, loc="upper right")
plt.show()

# Residuals tests

acorr_ljungbox(erros, lags=[1, 2, 3, 4, 5, 6, 7, 14, 21, 30], return_df=True)

"""
      lb_stat  lb_pvalue
1    0.058336   0.809146
2    0.071779   0.964747
3    0.089077   0.993115
4    0.097717   0.998845
5    0.442777   0.994069
6    1.676842   0.946895
7    1.871390   0.966614
14   6.320004   0.957770
21   9.139710   0.988072
30  22.370737   0.840220
"""

## SHAPIRO-WILK TEST

shapiro(erros)

# Shapiro Result
#            statistic |              pvalue
#   0.9763738643138354 | 0.49548958788392816

# p-value > 0.05 so we DO NOT REJECT the null hypothesis, so we can confirm residuals
# are normally distribuited
# Shapiro-Wilk shows that the residuals are normally distributed.

## JARQUE-BERA TEST

jarque_bera(erros)

# Significance Result
#            statistic |            pvalue
#   0.6522932049030084 |  0.72169937677797

# The p-value > 0.05 so we DO NOT REJECT the null hypothesis and confirm residuals DO
# FOLLOW A NORMAL DISTRIBUTION
# The Jarque-Bera test comproves the residuals are NORMALLY DISTRIBUTED

sns.set_palette("viridis")
plt.figure(figsize=(15, 10), dpi=300)
sns.lineplot(eur_train_arima["log"][1:], color="green", label="Valores reais (log)", linewidth=3)
sns.lineplot(eur_train_arima["fitted"][1:], color="magenta", label="Fitted Values", linewidth=3)
plt.title("Valores reais x Fitted Values - SARIMAX(2,1,2)(0,0,0)[0] + int.", fontsize="18")
plt.yticks(np.arange(round(eur_train_arima["log"].min() - 0.01, 2), round(eur_train_arima["log"].max() + 0.01, 2), 0.01), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d, %b'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Câmbio log(EUR ↔ BRL)", fontsize="18")
plt.legend(fontsize=18, loc="lower right", bbox_to_anchor=(0.99, 0.05, 0, 0))
plt.show()

# Revert Fitted Values:

eur_train_arima["invfit"] = invert_series(eur_train_arima["fitted"])

sns.set_palette("viridis")
plt.figure(figsize=(15, 10), dpi=300)
sns.lineplot(eur_train_arima["eur"][1:], color="green", label="Valores reais", linewidth=3)
sns.lineplot(eur_train_arima["invfit"][1:], color="magenta", label="Fitted Values", linewidth=3)
plt.title("Valores reais x Fitted Values - SARIMAX(2,1,2)(0,0,0)[0] + int.", fontsize="18")
plt.yticks(np.arange(round(eur_train_arima["eur"].min() - 0.1, 2), round(eur_train_arima["eur"].max() + 0.1, 2), 0.1), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d, %b'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Câmbio EUR ↔ BRL", fontsize="18")
plt.legend(fontsize=18, loc="lower right", bbox_to_anchor=(0.99, 0.05, 0, 0))
plt.show()

# In[2.0] Testing the ARIMA model with new data

df_eur_arima_3months_test = pd.read_csv("./datasets/arima_ready/eur_test_3months.csv", float_precision="high", parse_dates=([0]))

df_eur_arima_3months_test.info()

eur_test_arima = pd.DataFrame(df_eur_arima_3months_test.drop(["date"], axis=1).values, index=pd.date_range(start="2024-08-26", periods=20, freq="B"), columns=["eur", "log", "diff"])

y = eur_train_arima["log"]

# In[2.1]: Get predictted values

eur_test_arima = pd.concat([eur_test_arima, arima_3months_test.get_forecast(steps=20).summary_frame()], axis=1).reindex(eur_test_arima.index)

eur_arima_forecast = arima_3months_test.get_forecast(steps=20).summary_frame()

del eur_test_arima["forecast"]

eur_test_arima["inv_predict"] = invert_series(eur_test_arima["forecast"])

sns.set_palette("viridis_r")
plt.figure(figsize=(15, 10), dpi=300)
sns.lineplot(eur_train_arima["eur"][1:], label="Valores de Treino", linewidth=3)
sns.lineplot(eur_test_arima["eur"], label="Valores de Teste", linewidth=3)
sns.lineplot(eur_test_arima["inv_predict"], label="Valores preditos", linewidth=3)
plt.title("Valores reais x Fitted Values - SARIMAX(2,1,2)(0,0,0)[0] + int.", fontsize="18")
plt.yticks(np.arange(round(eur_train_arima["eur"].min() - 0.1, 2), round(eur_train_arima["eur"].max() + 0.1, 2), 0.1), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d, %b'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Câmbio EUR ↔ BRL", fontsize="18")
plt.legend(fontsize=18, loc="lower right", bbox_to_anchor=(0.99, 0.05, 0, 0))
plt.show()