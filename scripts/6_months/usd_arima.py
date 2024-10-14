# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # #
# Finding the ARIMA model = DOLLAR - 6 months #
# # # # # # # # # # # # # # # # # # # # # # # #

# In[0.1]: Importação dos pacotes

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.api import qqplot, qqline
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import shapiro, jarque_bera
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.stats.api import het_breuschpagan, het_arch
from itertools import repeat


# In[0.2]: Import dataframes

df_usd_arima_6months = pd.read_csv("./datasets/arima_ready/usd_train_6months.csv", float_precision="high", parse_dates=([0]))

df_usd_arima_6months.info()

usd_train_arima = pd.DataFrame(df_usd_arima_6months.drop(["date"], axis=1).values, index=pd.date_range(start="2024-03-25", periods=130, freq="B"), columns=["usd", "log", "diff", "eur"])

y = usd_train_arima["log"]

def invert_series(series):
    return np.exp(series)


sns.set_style("whitegrid",  {"grid.linestyle": ":"})
sns.set_palette("viridis")

# In[0.3] Define exogs:

# Normal dataset with normal exogs

usd_exog1 = pd.concat([usd_train_arima["eur"].shift(4)], axis=1).dropna()
usd_exog2 = pd.concat([usd_train_arima["eur"].shift(5)], axis=1).dropna()
usd_exog3 = pd.concat([usd_train_arima["eur"].shift(6)], axis=1).dropna()

len_usd_original = len(usd_train_arima)
len_usd_exog1 = len_usd_original - len(usd_exog1) - 1
len_usd_exog2 = len_usd_original - len(usd_exog2) - 1
len_usd_exog3 = len_usd_original - len(usd_exog3) - 1

df_usd_exog1 = usd_train_arima['usd'].drop(usd_train_arima['usd'].loc[0:len_usd_exog1].index)
df_usd_exog2 = usd_train_arima['usd'].drop(usd_train_arima['usd'].loc[0:len_usd_exog2].index)
df_usd_exog3 = usd_train_arima['usd'].drop(usd_train_arima['usd'].loc[0:len_usd_exog3].index)

# In[1.0]: Chosing and training the arima model

arima_6months_test = auto_arima(y,
                                seasonal = False,
                                stepwise = True,
                                trace = True).fit(y)

arima_6months_test.summary()

"""
Performing stepwise search to minimize aic
Best model:  ARIMA(0,1,0)(0,0,0)[0] - SARIMAX(0, 1, 0)
"""

arima_6months_test_010 = ARIMA(y, exog=None, order=(0,1,0)).fit()

arima_6months_test_010.summary()

arima_6months_test_011 = ARIMA(y, exog=None, order=(0,1,1)).fit()

arima_6months_test_011.summary()

# SARIMAX(0,1,0) LL 443 AIC -885 BIC -882
# SARIMAX(2,1,0) && (1,1,1) LL 444, AIC -883, BIC -874
# SARIMAX(1,1,0) && ARIMA(1,1,0) LL 444, AIC -885, BIC -879
# ARIMA(0,1,0) LL 443, AIC -885, BIC -882
# ARIMA(2,1,0) LL 444, AIC -883, BIC -874
# ARIMA(1,1,0) LL 444, AIC -885, BIC -879
# ARIMA (0,1,1) LL 444 AIC -885 BIC -879
# ARIMA(0,1,0) + linear trend LL 443 AIC -883 BIC -878
# ARIMA(1,1,0) + linear trend LL 444 AIC -883 BIC -875


"""
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                    log   No. Observations:                  130
Model:                 ARIMA(0, 1, 0)   Log Likelihood                 443.547
Date:                Fri, 04 Oct 2024   AIC                           -885.095
Time:                        02:45:04   BIC                           -882.235
Sample:                    03-25-2024   HQIC                          -883.933
                         - 09-20-2024                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
sigma2      6.037e-05   7.36e-06      8.206      0.000     4.6e-05    7.48e-05
===================================================================================
Ljung-Box (L1) (Q):                   1.91   Jarque-Bera (JB):                 0.05
Prob(Q):                              0.17   Prob(JB):                         0.98
Heteroskedasticity (H):               1.56   Skew:                             0.03
Prob(H) (two-sided):                  0.15   Kurtosis:                         3.08
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
"""

plt.figure(dpi=600)
fig = arima_6months_test_010.plot_diagnostics(figsize=(15,10), lags=30)
plt.suptitle("Diagnóstico dos Resíduos - ARIMA (0,1,0)", fontsize="18")
plt.show()

plt.figure(dpi=600)
fig = arima_6months_test_011.plot_diagnostics(figsize=(15,10), lags=30)
plt.suptitle("Diagnóstico dos Resíduos - ARIMA (0,1,1)", fontsize="18")
plt.show()

# In[1.1]: Saving and plotting the residuals for inspection

usd_train_arima["erros010"] = arima_6months_test_010.resid

usd_train_arima["fitted010"] = arima_6months_test_010.fittedvalues

erros_010 = usd_train_arima["erros010"][1:]

het_arch(erros_010)

"""
Engle's Test for Autoregressive Conditional Heteroskedasticity (ARCH)'
Lagrange statistic: 7.500347860853752
Lagrange p-value: 0.6775139316267902
F-statistic: 0.7264933597831472
F p-value: 0.6980348872088269

H0: Homoskedasticity
Hn: Heteroskedasticity

So, for the ARCH test, p-value > 0.05 which means we DO NOT REJECT H0, suggesting
the residuals are homoskedastic, meaning they have constant variance meaning the
model was well fitted.
"""

erros_010.describe()

"""
count    129.000000
mean       0.000629
std        0.007776
min       -0.021195
25%       -0.004146
50%        0.000362
75%        0.005064
max        0.019646
Name: erros, dtype: float64
"""

usd_train_arima["erros011"] = arima_6months_test_011.resid

usd_train_arima["fitted011"] = arima_6months_test_011.fittedvalues

new_df = usd_train_arima['log'] - usd_train_arima['erros010']

erros_011 = usd_train_arima["erros011"][1:]

erros_011.describe()

"""
count    129.000000
mean       0.000551
std        0.007718
min       -0.021503
25%       -0.004523
50%        0.000273
75%        0.004383
max        0.019803
Name: erros011, dtype: float64
"""

# Plot ACF and PACF of Residuals:

fig, (ax1, ax2) = plt.subplots(2, figsize=(11,8), dpi=300)
plot_acf(erros_010, ax=ax1)
plot_pacf(erros_010, ax=ax2)


fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(11,7), dpi=300)
plot_acf(erros_010, ax=ax1)
ax1.set_title("Autocorrelação - ARIMA (0,1,0)")
plot_pacf(erros_010, ax=ax3)
ax3.set_title("Autocorrelação Parcial - ARIMA (0,1,0)")
plot_acf(erros_011, ax=ax2)
ax2.set_title("Autocorrelação - ARIMA (0,1,1)")
plot_pacf(erros_011, ax=ax4)
ax4.set_title("Autocorrelação Parcial - ARIMA (0,1,1)")
for ax in [ax1, ax2, ax3]:
    ax.spines['bottom'].set(linewidth=1.5, color="black")
    ax.spines['left'].set(linewidth=1.5, color="black")


# Plot the scatter plot

plt.figure(figsize=(15, 10))
sns.scatterplot(erros_010)
plt.title("Resíduos do Modelo - ARIMA(0,1,0)", fontsize="18")
plt.xlabel("")
plt.ylabel("")
plt.legend(fontsize=18, loc="upper right")
plt.show()

# Residuals tests

acorr_ljungbox(erros_010**2, lags=[1, 2, 3, 4, 5, 6, 7, 14, 21, 30], return_df=True)

"""
      lb_stat  lb_pvalue
1    1.914660   0.166446
2    1.926192   0.381709
3    2.206547   0.530660
4    2.328697   0.675549
5    4.122396   0.531932
6    5.358264   0.498750
7    5.372018   0.614662
14  15.681034   0.333239
21  20.343664   0.499596
30  28.975251   0.518897
"""

acorr_ljungbox(erros_011, lags=[1, 2, 3, 4, 5, 6, 7, 14, 21, 30], return_df=True)

"""
      lb_stat  lb_pvalue
1    0.001727   0.966851
2    0.048728   0.975930
3    0.377397   0.944871
4    0.771110   0.942279
5    2.397482   0.791849
6    3.348113   0.764055
7    3.435226   0.842038
14  16.028888   0.311613
21  20.542689   0.487158
30  30.083022   0.461407
"""


## SHAPIRO-WILK TEST

shapiro(erros_010)

# Shapiro Result
#            statistic |             pvalue
#   0.9908359584294867 | 0.5576591680256284

# p-value > 0.05 so we DO NOT REJECT the null hypothesis, so we can confirm residuals
# are normally distribuited
# Shapiro-Wilk shows that the residuals are normally distributed.

shapiro(erros_011)

# Shapiro Result
#            statistic |              pvalue
#   0.9899967777616905 | 0.47925330565350244

# p-value > 0.05 so we DO NOT REJECT the null hypothesis, so we can confirm residuals
# are normally distribuited
# Shapiro-Wilk shows that the residuals are normally distributed.

## JARQUE-BERA TEST

jarque_bera(erros_010)

# Significance Result
#             statistic |              pvalue
#   0.04977461820656666 |  0.9754198267699749

# The p-value > 0.05 so we DO NOT REJECT the null hypothesis and confirm residuals DO
# FOLLOW A NORMAL DISTRIBUTION
# The Jarque-Bera test comproves the residuals are NORMALLY DISTRIBUTED

jarque_bera(erros_011)

# Significance Result
#             statistic |              pvalue
#   0.11627146453133369 |  0.9435218723835918

# The p-value > 0.05 so we DO NOT REJECT the null hypothesis and confirm residuals DO
# FOLLOW A NORMAL DISTRIBUTION
# The Jarque-Bera test comproves the residuals are NORMALLY DISTRIBUTED

sns.set_palette("viridis")
plt.figure(figsize=(15, 10), dpi=600)
sns.lineplot(usd_train_arima["log"], label="Valores reais (log)", linewidth=3)
sns.lineplot(usd_train_arima["fitted010"][1:], label="Fitted Values - 010", linewidth=3)
# plt.title("Valores reais x Fitted Values - USD - ARIMA(0,1,0)", fontsize="18")
plt.yticks(np.arange(round(usd_train_arima["log"].min() - 0.01, 2), round(usd_train_arima["log"].max() + 0.01, 2), 0.01), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d, %b'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=15))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Câmbio log(USD ↔ BRL)", fontsize="18")
plt.legend(fontsize=18, loc="lower right", bbox_to_anchor=(0.99, 0.05, 0, 0))
plt.savefig('./plots/save/valorrealxfittedvalues.tiff')
plt.show()

# Revert Fitted Values:

usd_train_arima["invfit010"] = invert_series(usd_train_arima["fitted010"])
usd_train_arima["invfit011"] = invert_series(usd_train_arima["fitted011"])

sns.set_palette("viridis")
plt.figure(figsize=(15, 10), dpi=300)
sns.lineplot(usd_train_arima["usd"][1:], label="Valores reais", linewidth=3)
sns.lineplot(usd_train_arima["invfit010"][1:], label="Fitted Values", linewidth=3)
sns.lineplot(usd_train_arima["invfit011"][1:], label="Fitted Values", linewidth=3)
# plt.title("Valores reais x Fitted Values - USD - ARIMA(0,1,0)", fontsize="18")
plt.yticks(np.arange(round(usd_train_arima["usd"].min() - 0.1, 2), round(usd_train_arima["usd"].max() + 0.1, 2), 0.1), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d, %b'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=15))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Câmbio USD ↔ BRL", fontsize="18")
plt.legend(fontsize=18, loc="lower right", bbox_to_anchor=(0.99, 0.05, 0, 0))
plt.show()

# In[2.0] Testing the ARIMA model with new data

df_usd_arima_6months_test = pd.read_csv("./datasets/arima_ready/usd_test_6months.csv", float_precision="high", parse_dates=([0]))

df_usd_arima_6months_test.info()

usd_test_arima = pd.DataFrame(df_usd_arima_6months_test.drop(["date"], axis=1).values[1:], index=pd.date_range(start="2024-09-23", periods=5, freq="B"), columns=["usd", "log", "diff"])

# In[2.1]: Get predictted values

usd_forecast_arima = pd.concat([usd_test_arima, invert_series(arima_6months_test_010.get_forecast(steps=5).summary_frame())], axis=1).reindex(usd_test_arima.index)

usd_forecast_arima = usd_forecast_arima.rename(columns={"mean": "mean010", "mean_se": "mean_se010", "mean_ci_lower": "mean_ci_lower010", "mean_ci_upper": "mean_ci_upper010"})

usd_forecast_arima = pd.concat([usd_forecast_arima, invert_series(arima_6months_test_011.get_forecast(steps=5).summary_frame())], axis=1)

usd_forecast_arima = usd_forecast_arima.rename(columns={"mean": "mean011", "mean_se": "mean_se011", "mean_ci_lower": "mean_ci_lower011", "mean_ci_upper": "mean_ci_upper011"})

usd_arima_forecast = arima_6months_test.get_forecast(steps=5).summary_frame()


sns.set_palette("viridis")
fig, ax = plt.subplots(figsize=(15, 10), dpi=300)
sns.lineplot(usd_test_arima["usd"], label="Valores de Teste", linewidth=3)
sns.lineplot(usd_forecast_arima["mean010"], label="Valores preditos - 010", linewidth=3)
sns.lineplot(usd_forecast_arima["mean011"], label="Valores preditos - 011", linewidth=3)
# plt.title("Valores reais x Fitted Values", fontsize="18")
plt.yticks(np.arange(round(usd_forecast_arima["usd"].min() - 0.1, 2), round(usd_forecast_arima["usd"].max() + 0.1, 2), 0.1), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d, %b'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Câmbio USD ↔ BRL", fontsize="18")
plt.legend(fontsize=18, loc="lower right", bbox_to_anchor=(0.99, 0.05, 0, 0))
plt.show()


sns.set_palette("viridis")
fig, ax = plt.subplots(figsize=(15, 10), dpi=300)
sns.lineplot(usd_train_arima["usd"][1:], label="Valores de Treino", linewidth=3)
sns.lineplot(usd_test_arima["usd"], label="Valores de Teste", linewidth=3)
sns.lineplot(usd_test_arima["mean"], label="Valores preditos", linewidth=3)
plt.title("Valores reais x Fitted Values - USD - ARIMA(0,1,0)", fontsize="18")
plt.yticks(np.arange(round(usd_train_arima["usd"].min() - 0.1, 2), round(usd_train_arima["usd"].max() + 0.1, 2), 0.1), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d, %b'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=15))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Câmbio USD ↔ BRL", fontsize="18")
plt.legend(fontsize=18, loc="lower right", bbox_to_anchor=(0.99, 0.05, 0, 0))
plt.show()

sns.set_palette("viridis")
fig, ax = plt.subplots(figsize=(15, 10), dpi=300)
sns.lineplot(usd_test_arima["usd"], label="Valores de Teste", linewidth=3)
sns.lineplot(usd_test_arima["mean"], label="Valores preditos", linewidth=3)
ax.fill_between(usd_test_arima.index, usd_test_arima["mean_ci_lower"], usd_test_arima["mean_ci_upper"], alpha=0.15, label="Intervalo de confiança")
plt.title("Valores reais x Fitted Values - USD - ARIMA(0,1,0)", fontsize="18")
plt.yticks(np.arange(round(usd_test_arima["mean_ci_lower"].min() - 0.01, 2), round(usd_test_arima["mean_ci_upper"].max() + 0.01, 2), 0.03), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d, %b'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Câmbio USD ↔ BRL", fontsize="18")
plt.legend(fontsize=18, loc="upper right")
plt.show()

# In[3.0]: Testing other automated tests:

arima_6months_auto = ARIMA(y, exog=None, order=(1,1,0), enforce_stationarity=True).fit()

arima_6months_auto.summary()

usd_train_arima["errosAuto"] = arima_6months_auto.resid

usd_train_arima["fittedAuto"] = arima_6months_auto.fittedvalues

errosAuto = usd_train_arima["errosAuto"][1:]

errosAuto.describe()
"""
count    129.000000
mean       0.000542
std        0.007719
min       -0.021650
25%       -0.004658
50%        0.000215
75%        0.004329
max        0.019551
Name: errosAuto, dtype: float64
"""

shapiro(errosAuto)
# ShapiroResult(statistic=0.9897125320880724, pvalue=0.45420463888225204)

jarque_bera(errosAuto)
# SignificanceResult(statistic=0.10567494454119607, pvalue=0.9485341630588694)

usd_arima_forecast_auto = invert_series(arima_6months_auto.get_forecast(steps=5).summary_frame())

sns.set_palette("viridis")
fig, ax = plt.subplots(1, figsize=(15, 10), dpi=600)
sns.lineplot(usd_train_arima["usd"][1:], label="Valores de Treino", linewidth=3)
sns.lineplot(usd_test_arima["usd"], label="Valores de Teste", linewidth=3)
sns.lineplot(usd_test_arima["mean"], label="Valores preditos (manual)", linewidth=3)
sns.lineplot(usd_arima_forecast_auto["mean"], label="Valores preditos (auto)", linewidth=3)
plt.title("Valores reais x Previsões - USD -  ARIMA(1,1,1)", fontsize="18")
plt.yticks(np.arange(round(usd_train_arima["usd"].min() - 0.05, 2), round(usd_train_arima["usd"].max() + 0.05, 2), 0.05), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d, %b'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=15))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Câmbio USD ↔ BRL", fontsize="18")
plt.legend(fontsize=18, loc="upper left")
plt.show()


exponential_usd = ExponentialSmoothing(usd_train_arima["usd"][-5:], freq="B").fit()

exponential_usd.summary()


sns.set_palette("viridis")
fig, ax = plt.subplots(figsize=(15, 10), dpi=600)
sns.lineplot(usd_test_arima["usd"], label="Valores de Teste", linewidth=2)
sns.lineplot(usd_test_arima["mean"], label="Valores preditos ARIMA(0,1,0)", lw=2)
sns.lineplot(y=usd_train_arima[-5:]["usd"], x=usd_test_arima.index, label="Predição Naive Sazonal", lw=3, linestyle="--")
sns.lineplot(y=list(repeat(np.mean(usd_train_arima[-5:]["usd"]), 5)), x=usd_test_arima.index, label="Predição de Janela Móvel(5)", lw=3, linestyle="--")
sns.lineplot(y=exponential_usd.fittedvalues, x=usd_test_arima.index, label="Suavização exponencial simples", lw=3, linestyle="--")
ax.fill_between(usd_test_arima.index, usd_test_arima["mean_ci_lower"], usd_test_arima["mean_ci_upper"], alpha=0.15, label="Intervalo de confiança ARIMA(0,1,0)")
# plt.title("Valores reais x Fitted Values - USD - ARIMA(0,1,0)", fontsize="18")
plt.yticks(np.arange(round(usd_test_arima["mean_ci_lower"].min() - 0.01, 2), round(usd_test_arima["mean_ci_upper"].max() + 0.01, 2), 0.03), fontsize="22")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d, %b'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="18")
plt.ylabel("Câmbio USD ↔ BRL", fontsize="22")
plt.xlabel("")
plt.legend(fontsize="22", loc="center", bbox_to_anchor=(0.5, 0.5,1.6,0) )
plt.show()

# Plotting wiht the prediction average:

predict_average = pd.DataFrame({
    "arima": usd_test_arima["mean"],
    "exponential": exponential_usd.fittedvalues.values,
    "window_avg": list(repeat(np.mean(usd_train_arima["usd"][-5:]), 5)),
    "naive": usd_train_arima["usd"][-5:].values,
    }, index=usd_test_arima.index)

predict_average["mean"] = predict_average.mean(axis=1)

sns.set_palette("viridis")
fig, ax = plt.subplots(figsize=(15, 10), dpi=600)
sns.lineplot(usd_test_arima["usd"], label="Valores de Teste", linewidth=3)
sns.lineplot(usd_test_arima["mean"], label="Valores preditos ARIMA(0,1,0)", lw=3)
sns.lineplot(y=usd_train_arima[-5:]["usd"], x=usd_test_arima.index, label="Predição Naive Sazonal", lw=3)
sns.lineplot(y=list(repeat(np.mean(usd_train_arima[-5:]["usd"]), 5)), x=usd_test_arima.index, label="Predição Média Móvel(5)", lw=3)
sns.lineplot(y=exponential_usd.fittedvalues, x=usd_test_arima.index, label="Suavização exponencial simples", lw=3)
sns.lineplot(predict_average["mean"], label="Média das previsões dos modelos", lw=4, linestyle="--", color="darkorchid")
ax.fill_between(usd_test_arima.index, usd_test_arima["mean_ci_lower"], usd_test_arima["mean_ci_upper"], alpha=0.15, label="Intervalo de confiança ARIMA(0,1,0)")
# plt.title("Valores reais x Fitted Values - USD - ARIMA(0,1,0)", fontsize="18")
plt.yticks(np.arange(round(usd_test_arima["mean_ci_lower"].min() - 0.01, 2), round(usd_test_arima["mean_ci_upper"].max() + 0.01, 2), 0.03), fontsize="22")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d, %b'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="18")
plt.ylabel("Câmbio USD ↔ BRL", fontsize="22")
plt.xlabel("")
plt.legend(fontsize="22", loc="center", bbox_to_anchor=(0.5, 0.5,1.6,0) )
plt.show()


sns.set_palette("viridis")
fig, ax = plt.subplots(figsize=(15, 10), dpi=600)
sns.lineplot(usd_test_arima["usd"], label="Valores de Teste", linewidth=3)
sns.lineplot(usd_test_arima["mean"], label="Valores preditos ARIMA(0,1,0)", lw=3)
sns.lineplot(predict_average["mean"], label="Média das previsões dos modelos", lw=4, linestyle="--", color="darkorchid")
ax.fill_between(usd_test_arima.index, usd_test_arima["mean_ci_lower"], usd_test_arima["mean_ci_upper"], alpha=0.15, label="Intervalo de confiança ARIMA(0,1,0)")
# plt.title("Valores reais x Fitted Values - USD - ARIMA(0,1,0)", fontsize="18")
plt.yticks(np.arange(round(usd_test_arima["mean_ci_lower"].min() - 0.01, 2), round(usd_test_arima["mean_ci_upper"].max() + 0.01, 2), 0.03), fontsize="22")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d, %b'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="18")
plt.ylabel("Câmbio USD ↔ BRL", fontsize="22")
plt.xlabel("")
plt.legend(fontsize="22", loc="lower right")
plt.show()

# In[]: Calculating the RMSE for each model:

# Actual values (true data)
y_true = np.array([1.5, 2.0, 3.5, 4.0, 5.5])

# Predicted values from your model
y_pred = np.array([1.4, 2.1, 3.6, 4.2, 5.4])

# Calculate RMSE manually
rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
print("RMSE:", rmse)