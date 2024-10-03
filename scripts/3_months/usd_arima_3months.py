# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # #
# Finding the ARIMA model = DOLLAR - 3 months #
# # # # # # # # # # # # # # # # # # # # # # # #

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

df_usd_arima_3months = pd.read_csv("./datasets/arima_ready/usd_train_3months.csv", float_precision="high", parse_dates=([0]))

df_usd_arima_3months.info()

usd_train_arima = pd.DataFrame(df_usd_arima_3months.drop(["date"], axis=1).values, index=pd.date_range(start="2024-06-24", periods=45, freq="B"), columns=["usd", "log", "diff", "lag2"])

y = usd_train_arima["log"]

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
 ARIMA(2,2,2)(0,0,0)[0] intercept   : AIC=-267.001, Time=0.33 sec
 ARIMA(0,2,0)(0,0,0)[0] intercept   : AIC=-259.736, Time=0.01 sec
 ARIMA(1,2,0)(0,0,0)[0] intercept   : AIC=-265.692, Time=0.02 sec
 ARIMA(0,2,1)(0,0,0)[0] intercept   : AIC=-271.646, Time=0.13 sec
 ARIMA(0,2,0)(0,0,0)[0]             : AIC=-261.603, Time=0.02 sec
 ARIMA(1,2,1)(0,0,0)[0] intercept   : AIC=-267.494, Time=0.08 sec
 ARIMA(0,2,2)(0,0,0)[0] intercept   : AIC=-271.258, Time=0.11 sec
 ARIMA(1,2,2)(0,0,0)[0] intercept   : AIC=-266.690, Time=0.24 sec
 ARIMA(0,2,1)(0,0,0)[0]             : AIC=-273.243, Time=0.05 sec
 ARIMA(1,2,1)(0,0,0)[0]             : AIC=-273.594, Time=0.08 sec
 ARIMA(1,2,0)(0,0,0)[0]             : AIC=-267.577, Time=0.03 sec
 ARIMA(2,2,1)(0,0,0)[0]             : AIC=inf, Time=0.12 sec
 ARIMA(1,2,2)(0,0,0)[0]             : AIC=inf, Time=0.16 sec
 ARIMA(0,2,2)(0,0,0)[0]             : AIC=inf, Time=0.14 sec
 ARIMA(2,2,0)(0,0,0)[0]             : AIC=-266.873, Time=0.07 sec
 ARIMA(2,2,2)(0,0,0)[0]             : AIC=-269.902, Time=0.18 sec

Best model:  ARIMA(1,2,1)(0,0,0)[0]          
Total fit time: 1.769 seconds
"""

arima_3months_test = SARIMAX(y, exog=None, order=(2,1,1), seasonal_order=(0,0,0,0), enforce_stationarity=True, trend="c").fit()

arima_3months_test.summary()

"""
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                    log   No. Observations:                   45
Model:               SARIMAX(2, 1, 1)   Log Likelihood                 145.715
Date:                Mon, 30 Sep 2024   AIC                           -283.430
Time:                        22:18:05   BIC                           -276.293
Sample:                    06-24-2024   HQIC                          -280.783
                         - 08-23-2024                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ar.L1         -0.4199      0.803     -0.523      0.601      -1.995       1.155
ar.L2          0.2449      0.202      1.213      0.225      -0.151       0.641
ma.L1          0.6059      0.788      0.769      0.442      -0.939       2.151
sigma2      7.755e-05   2.07e-05      3.743      0.000    3.69e-05       0.000
===================================================================================
Ljung-Box (L1) (Q):                   0.02   Jarque-Bera (JB):                 0.30
Prob(Q):                              0.89   Prob(JB):                         0.86
Heteroskedasticity (H):               0.76   Skew:                             0.04
Prob(H) (two-sided):                  0.61   Kurtosis:                         2.60
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
"""

arima_3months_test.plot_diagnostics(figsize=(10,7))
plt.show()

# In[1.1]: Saving and plotting the residuals for inspection

usd_train_arima["erros"] = arima_3months_test.resid

usd_train_arima["fitted"] = arima_3months_test.fittedvalues

erros = usd_train_arima["erros"][1:]

erros.describe()

"""
count    44.000000
mean      0.000040
std       0.008973
min      -0.023198
25%      -0.005510
50%      -0.000248
75%       0.005299
max       0.017789
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
plt.title("Resíduos do Modelo - ARIMA(2,1,1) + intercepto", fontsize="18")
plt.xlabel("")
plt.ylabel("")
plt.legend(fontsize=18, loc="upper right")
plt.show()

# Residuals tests

acorr_ljungbox(erros, lags=[1, 2, 3, 4, 5, 6, 7, 14, 21, 30], return_df=True)

"""
      lb_stat  lb_pvalue
1    0.003694   0.951538
2    0.005223   0.997392
3    0.600562   0.896304
4    1.287297   0.863527
5    1.602610   0.900934
6    2.624234   0.854313
7    2.788337   0.903868
14  10.694198   0.709875
21  11.678301   0.947898
30  24.700580   0.739448
"""

## SHAPIRO-WILK TEST

shapiro(erros)

# Shapiro Result
#           statistic |             pvalue
#   0.982565014231839 | 0.7362568643897442

# p-value > 0.05 so we DO NOT REJECT the null hypothesis, so we can confirm residuals
# are normally distribuited
# Shapiro-Wilk shows that the residuals are normally distributed.

## JARQUE-BERA TEST

jarque_bera(erros)

# Significance Result
#             statistic |            pvalue
#   0.06478711565333846 |  0.96812549372871

# The p-value > 0.05 so we DO NOT REJECT the null hypothesis and confirm residuals DO
# FOLLOW A NORMAL DISTRIBUTION
# The Jarque-Bera test comproves the residuals are NORMALLY DISTRIBUTED

sns.set_palette("viridis")
plt.figure(figsize=(15, 10), dpi=300)
sns.lineplot(usd_train_arima["log"][1:], color="green", label="Valores reais (log)", linewidth=3)
sns.lineplot(usd_train_arima["fitted"][1:], color="magenta", label="Fitted Values", linewidth=3)
plt.title("Valores reais x Fitted Values - SARIMAX(2,1,2)(0,0,0)[0] + int.", fontsize="18")
plt.yticks(np.arange(round(usd_train_arima["log"].min() - 0.01, 2), round(usd_train_arima["log"].max() + 0.01, 2), 0.01), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d, %b'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Câmbio log(USD ↔ BRL)", fontsize="18")
plt.legend(fontsize=18, loc="lower right", bbox_to_anchor=(0.99, 0.05, 0, 0))
plt.show()

# Revert Fitted Values:

usd_train_arima["invfit"] = invert_series(usd_train_arima["fitted"])

sns.set_palette("viridis")
plt.figure(figsize=(15, 10), dpi=300)
sns.lineplot(usd_train_arima["usd"][1:], color="green", label="Valores reais", linewidth=3)
sns.lineplot(usd_train_arima["invfit"][1:], color="magenta", label="Fitted Values", linewidth=3)
plt.title("Valores reais x Fitted Values - SARIMAX(2,1,2)(0,0,0)[0] + int.", fontsize="18")
plt.yticks(np.arange(round(usd_train_arima["usd"].min() - 0.1, 2), round(usd_train_arima["usd"].max() + 0.1, 2), 0.1), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d, %b'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Câmbio USD ↔ BRL", fontsize="18")
plt.legend(fontsize=18, loc="lower right", bbox_to_anchor=(0.99, 0.05, 0, 0))
plt.show()

# In[2.0] Testing the ARIMA model with new data

df_usd_arima_3months_test = pd.read_csv("./datasets/arima_ready/usd_test_3months.csv", float_precision="high", parse_dates=([0]))

df_usd_arima_3months_test.info()

usd_test_arima = pd.DataFrame(df_usd_arima_3months_test.drop(["date"], axis=1).values, index=pd.date_range(start="2024-08-26", periods=20, freq="B"), columns=["usd", "log", "diff"])

y = usd_train_arima["log"]

# In[2.1]: Get predictted values

usd_test_arima = pd.concat([usd_test_arima, invert_series(arima_3months_test.get_forecast(steps=20).summary_frame())], axis=1).reindex(usd_test_arima.index)

usd_arima_forecast = invert_series(arima_3months_test.get_forecast(steps=20).summary_frame())

plt.style.use("seaborn-v0_8-colorblind")
plt.figure(figsize=(15, 10), dpi=300)
sns.lineplot(usd_train_arima["usd"][1:], label="Valores de Treino", linewidth=3)
sns.lineplot(usd_test_arima["usd"], label="Valores de Teste", linewidth=3)
sns.lineplot(usd_test_arima["mean"], label="Valores preditos", linewidth=3)
plt.title("Valores reais x Fitted Values - SARIMAX(2,1,1)(0,0,0)[0] + int.", fontsize="18")
plt.yticks(np.arange(round(usd_train_arima["usd"].min() - 0.1, 2), round(usd_train_arima["usd"].max() + 0.1, 2), 0.1), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d, %b'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Câmbio USD ↔ BRL", fontsize="18")
plt.legend(fontsize=18, loc="lower right", bbox_to_anchor=(0.99, 0.05, 0, 0))
plt.show()

plt.style.use("seaborn-v0_8-colorblind")
plt.figure(figsize=(15, 10), dpi=300)
sns.lineplot(usd_test_arima["eur"], label="Valores de Teste", linewidth=3)
sns.lineplot(usd_test_arima["mean"], label="Valores preditos", linewidth=3)
plt.title("Valores reais x Fitted Values - SARIMAX(0,1,0)(0,0,0)[0] + int.", fontsize="18")
plt.yticks(np.arange(round(usd_test_arima["eur"].min() - 0.01, 2), round(usd_test_arima["eur"].max() + 0.01, 2), 0.01), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d, %b'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Câmbio EUR ↔ BRL", fontsize="18")
plt.legend(fontsize=18, loc="upper right")
plt.show()