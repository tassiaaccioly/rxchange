# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # #
# Finding the ARIMA model = EURO - 6 months #
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

df_eur_arima_6months = pd.read_csv("./datasets/arima_ready/eur_train_6months.csv", float_precision="high", parse_dates=([0]))

df_eur_arima_6months.info()

eur_train_arima = pd.DataFrame(df_eur_arima_6months.drop(["date"], axis=1).values, index=pd.date_range(start="2024-03-25", periods=130, freq="B"), columns=["eur", "log", "diff"])

y = eur_train_arima["log"]

def invert_series(series):
    return np.exp(series)

# In[1.0]: Chosing and training the arima model

arima_6months_test = auto_arima(y,
                                test="adf",
                                seasonal = False,
                                stepwise = True,
                                trace = True).fit(y)

arima_6months_test.summary()

"""
Performing stepwise search to minimize aic
 ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=-897.287, Time=0.22 sec
 ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=-902.659, Time=0.02 sec
 ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=-902.086, Time=0.04 sec
 ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=-902.136, Time=0.09 sec
 ARIMA(0,1,0)(0,0,0)[0]             : AIC=-902.757, Time=0.01 sec
 ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=-900.135, Time=0.10 sec

Best model:  ARIMA(0,1,0)(0,0,0)[0]          
Total fit time: 0.485 seconds
"""

arima_6months_test = SARIMAX(y, exog=None, order=(0,1,0), seasonal_order=(0,0,0,0), enforce_stationarity=True, trend="c").fit()

arima_6months_test.summary()

"""
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                    log   No. Observations:                  130
Model:               SARIMAX(0, 1, 0)   Log Likelihood                 453.330
Date:                Tue, 01 Oct 2024   AIC                           -902.659
Time:                        02:31:24   BIC                           -896.940
Sample:                    03-25-2024   HQIC                          -900.335
                         - 09-20-2024                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
intercept      0.0009      0.001      1.383      0.167      -0.000       0.002
sigma2       5.19e-05      6e-06      8.654      0.000    4.01e-05    6.37e-05
===================================================================================
Ljung-Box (L1) (Q):                   1.46   Jarque-Bera (JB):                 0.71
Prob(Q):                              0.23   Prob(JB):                         0.70
Heteroskedasticity (H):               1.71   Skew:                             0.08
Prob(H) (two-sided):                  0.08   Kurtosis:                         3.33
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
"""

arima_6months_test.plot_diagnostics(figsize=(10,7))
plt.show()

# In[1.1]: Saving and plotting the residuals for inspection

eur_train_arima["erros"] = arima_6months_test.resid

eur_train_arima["fitted"] = arima_6months_test.fittedvalues

erros = eur_train_arima["erros"][1:]

erros.describe()

"""
count    1.290000e+02
mean     5.852338e-17
std      7.232064e-03
min     -1.827246e-02
25%     -4.005175e-03
50%      1.025687e-04
75%      3.892901e-03
max      1.817867e-02
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
1    1.457744   0.227289
2    1.541546   0.462655
3    3.945127   0.267451
4    4.290065   0.368171
5    5.320285   0.378055
6    6.815161   0.338280
7    6.870578   0.442479
14  10.622934   0.715370
21  17.388449   0.687307
30  24.192000   0.763269
"""

## SHAPIRO-WILK TEST

shapiro(erros)

# Shapiro Result
#            statistic |              pvalue
#   0.9841368175836072 | 0.13806235242836296

# p-value > 0.05 so we DO NOT REJECT the null hypothesis, so we can confirm residuals
# are normally distribuited
# Shapiro-Wilk shows that the residuals are normally distributed.

## JARQUE-BERA TEST

jarque_bera(erros)

# Significance Result
#            statistic |            pvalue
#   0.7090364024613426 |  0.7015113491053824

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
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=15))
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
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=15))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Câmbio EUR ↔ BRL", fontsize="18")
plt.legend(fontsize=18, loc="lower right", bbox_to_anchor=(0.99, 0.05, 0, 0))
plt.show()

# In[2.0] Testing the ARIMA model with new data

df_eur_arima_6months_test = pd.read_csv("./datasets/arima_ready/eur_test_6months.csv", float_precision="high", parse_dates=([0]))

df_eur_arima_6months_test.info()

eur_test_arima = pd.DataFrame(df_eur_arima_6months_test.drop(["date"], axis=1).values, index=pd.date_range(start="2024-09-23", periods=6, freq="B"), columns=["eur", "log", "diff"])

# In[2.1]: Get predictted values

eur_test_arima = pd.concat([eur_test_arima, invert_series(arima_6months_test.get_forecast(steps=6).summary_frame())], axis=1).reindex(eur_test_arima.index)

eur_arima_forecast = arima_6months_test.get_forecast(steps=6).summary_frame()

plt.style.use("seaborn-v0_8-colorblind")
fig, ax = plt.subplots(figsize=(15, 10), dpi=300)
sns.lineplot(eur_train_arima["eur"][1:], label="Valores de Treino", linewidth=3)
sns.lineplot(eur_test_arima["eur"], label="Valores de Teste", linewidth=3)
sns.lineplot(eur_test_arima["mean"], label="Valores preditos", linewidth=3)
plt.title("Valores reais x Fitted Values - SARIMAX(0,1,0)(0,0,0)[0] + int.", fontsize="18")
plt.yticks(np.arange(round(eur_train_arima["eur"].min() - 0.1, 2), round(eur_train_arima["eur"].max() + 0.1, 2), 0.1), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d, %b'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=15))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Câmbio EUR ↔ BRL", fontsize="18")
plt.legend(fontsize=18, loc="lower right", bbox_to_anchor=(0.99, 0.05, 0, 0))
plt.show()

plt.style.use("seaborn-v0_8-colorblind")
fig, ax = plt.subplots(figsize=(15, 10), dpi=300)
sns.lineplot(eur_test_arima["eur"], label="Valores de Teste", linewidth=3)
sns.lineplot(eur_test_arima["mean"], label="Valores preditos", linewidth=3)
ax.fill_between(eur_test_arima.index, eur_test_arima["mean_ci_lower"], eur_test_arima["mean_ci_upper"], alpha=0.15, label="Intervalo de confiança")
plt.title("Valores reais x Fitted Values - SARIMAX(0,1,0)(0,0,0)[0] + int.", fontsize="18")
plt.yticks(np.arange(round(eur_test_arima["mean_ci_lower"].min() - 0.01, 2), round(eur_test_arima["mean_ci_upper"].max() + 0.01, 2), 0.03), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d, %b'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Câmbio EUR ↔ BRL", fontsize="18")
plt.legend(fontsize=18, loc="upper right")
plt.show()