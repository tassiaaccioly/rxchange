# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # #
# Exploratory Data Analysis - DOLLAR - 3 months #
# # # # # # # # # # # # # # # # # # # # # # # # #

# In[0.1]: Importação dos pacotes

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from itertools import repeat
from scipy.stats import boxcox, norm, kruskal, mannwhitneyu, f_oneway
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.model_selection import TimeSeriesSplit
from statistics import mean, stdev
from scipy.stats import norm, shapiro, jarque_bera


# In[0.2]: Import dataframes

df_usd_1year = pd.read_csv("./datasets/wrangled/df_usd_1year.csv", float_precision="high", parse_dates=([1]))

df_usd_1year.info()

df_wg_usd_6months = df_usd_1year

df_wg_usd_6months.head()

dt_format = "%d/%m/%Y"

df_wg_usd_6months = df_wg_usd_6months.drop(df_wg_usd_6months[df_wg_usd_6months["weekday"] == "Sunday"].index)

df_wg_usd_6months = df_wg_usd_6months.drop(df_wg_usd_6months[df_wg_usd_6months["weekday"] == "Saturday"].index)[131:]

df_wg_usd_6months = df_wg_usd_6months.drop(["weekday", "date"], axis=1)

df_wg_usd_6months = pd.DataFrame(df_wg_usd_6months.drop("dateTime", axis=1).values, index=pd.date_range(start="2024-03-25", periods=130 ,freq="B"), columns=["usd", "log", "diff"])

df_usd_10days = pd.read_csv('./datasets/wrangled/df_usd_10days.csv', float_precision="high", parse_dates=([1]))

df_wg_usd_10days = df_usd_10days

df_wg_usd_10days = df_wg_usd_10days.drop(df_wg_usd_10days[df_wg_usd_10days["weekday"] == "Sunday"].index)

df_wg_usd_10days = df_wg_usd_10days.drop(df_wg_usd_10days[df_wg_usd_10days["weekday"] == "Saturday"].index)

df_wg_usd_10days = df_wg_usd_10days.drop(["weekday", "date"], axis=1)

df_wg_usd_10days = pd.DataFrame(df_wg_usd_10days.drop("dateTime", axis=1).values, index=pd.date_range(start="2024-09-23", periods=6 ,freq="B"), columns=["usd", "log", "diff"])

# In[0.3]: Separate data into train and test

# remove last 15 days of data for test later
usd_train = df_wg_usd_6months

usd_test = df_wg_usd_10days

# In[0.4]: Plot original datasets

# 3 months dataset
sns.set_style("whitegrid", {"grid.linestyle": ":"})
sns.set_palette("viridis")
fig, ax = plt.subplots(1, figsize=(15, 10), dpi=600)
ax.spines['bottom'].set(linewidth=3, color="black")
ax.spines['left'].set(linewidth=3, color="black")
sns.lineplot(usd_train["usd"], label="Câmbio USD - Treino", linewidth=3)
sns.lineplot(y=list(repeat(np.max(usd_train["usd"]), len(usd_train))), x=usd_train.index, lw=3, linestyle="--", label="Máxima") # max usd
sns.lineplot(y=list(repeat(np.mean(usd_train["usd"]), len(usd_train))), x=usd_train.index, lw=3, linestyle="--", label="Média") # mean usd
sns.lineplot(y=list(repeat(np.min(usd_train["usd"]), len(usd_train))), x=usd_train.index, lw=3, linestyle="--", label="Mínima") # min usd
# plt.title(f'Cotação do Dólar - Série histórica ({usd_train.index[0].strftime(dt_format)} - {usd_train.index[-1].strftime(dt_format)})', fontsize="18")
plt.yticks(np.arange(4.95, round(usd_train["usd"].max(), 1), 0.05), fontsize="22")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d, %b'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=15))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="22")
plt.xlabel("", fontsize="22")
plt.ylabel("Câmbio USD ↔ BRL", fontsize="22")
plt.legend(fontsize="22", loc="lower right", bbox_to_anchor=(0.98,0.05,0,0))
plt.savefig('./plots/save/hist_usd_original.tiff')
plt.show()

# Boxplot for 3months:

plt.figure(figsize=(15,10), dpi=300)
sns.boxplot(x=usd_train["usd"], palette="viridis")
plt.title("Boxplot do Dataset - 6 meses (130 dias)", fontsize="18")
plt.legend(fontsize="16")
plt.show()


plt.figure(figsize=(15,10), dpi=300)
sns.boxplot(x=usd_train["diff"], palette="viridis")
plt.title("Boxplot do Dataset - 6 meses (130 dias)", fontsize="18")
plt.legend(fontsize="16")
plt.show()

###############################################################################

# Plot "normal" dos dados
sns.set_palette("viridis")
fig, ax = plt.subplots(1, figsize=(15,10), dpi=600)
mn = mean(df_usd_1year["diff"].dropna())
strd = stdev(df_usd_1year["diff"].dropna())
# Get parameters for the normal curve
x_pdf = np.linspace(-0.1, 0.1, 300)
y_pdf = norm.pdf(x_pdf, mn, strd)
ax.spines["left"].set(lw=3, color="black")
ax.spines["bottom"].set(lw=3, color="black")
sns.histplot(df_usd_1year["diff"].dropna(), stat="density", kde="true", label="Dados com log + d(1)", line_kws={"label":"Est. de Densidade Kernel", "lw": 3})
ax.plot(x_pdf, y_pdf, lw=3, ls="--", label="Curva Normal")
plt.xticks(fontsize="22")
plt.yticks(fontsize="22")
plt.ylabel("Densidade", fontsize="22")
plt.xlabel("Dados", fontsize="22")
plt.legend(fontsize="22", loc="upper left")
plt.savefig('./plots/save/figura1.tiff', dpi=600, format="tiff", bbox_inches="tight")
plt.show()

###############################################################################

shapiro(df_wg_usd_6months["diff"].dropna())
# ShapiroResult(statistic=0.9361729706977153, pvalue=2.191773719359413e-11)

jarque_bera(df_wg_usd_6months["diff"].dropna())
# SignificanceResult(statistic=64.15126089437189, pvalue=1.1741692247184742e-14)

# In[0.5]: Plot "normal" dos dados cruzados

# Curva normal para os dados originais
mn1 = mean(df_usd_1year["usd"].dropna())
strd1 = stdev(df_usd_1year["usd"].dropna())
# Get parameters for the normal curve
x_pdf1 = np.linspace(4.7, 5.7, 300)
y_pdf1 = norm.pdf(x_pdf1, mn1, strd1)

# curva normal para os dados com log e diff
mn2 = mean(df_usd_1year["diff"].dropna())
strd2 = stdev(df_usd_1year["diff"].dropna())
# Get parameters for the normal curve
x_pdf2 = np.linspace(-0.1, 0.1, 300)
y_pdf2 = norm.pdf(x_pdf2, mn2, strd2)

sns.set_palette("viridis")
fig, ((ax1, ax3),(ax2, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(15,10), dpi=600)
plt.rcParams.update({"font.size": 22})

sns.histplot(df_usd_1year["usd"], stat="density", kde="true", label="Dados Originais", line_kws={"label":"Est. de Densidade Kernel", "lw": 3}, ax=ax1)
ax1.plot(x_pdf1, y_pdf1, lw=3, ls="--", label="Curva Normal")
sns.histplot(df_usd_1year["diff"].dropna(), stat="density", kde="true", label="Dados com log + d(1)", line_kws={"label":"Est. de Densidade Kernel", "lw": 3}, ax=ax3)
ax3.plot(x_pdf2, y_pdf2, lw=3, ls="--", label="Curva Normal")

for i, ax in enumerate([ax1, ax3, ax2, ax4]):
    ax.set_ylabel("Densidade", fontsize="16")
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize="16")
    ax.tick_params(axis="y", labelsize="16")
    ax.spines["left"].set(lw=3, color="black")
    ax.spines["bottom"].set(lw=3, color="black")
    ax.legend(fontsize="14", loc="upper left")
    ax.set_title(f"Gráfico {i+1}:", fontsize="18", loc="left")

plt.tight_layout()
plt.show()
# In[0.5]: Calculate Statistics for datasets

var_usd_3months = np.var(usd_train['usd'])
var_usd_3months

# 0.04764066744378699

varlog_usd_3months = np.var(usd_train['log'])
varlog_usd_3months

# 0.0016699954864482114

optimal_lags_usd_3months = 12*(len(usd_train['usd'])/100)**(1/4)
optimal_lags_usd_3months

# 12.813479668469292

# So optimal lags for this dataset = 9 or 10

# In[0.6]: Plot train and test datasets:

plt.figure(figsize=(15, 10), dpi=300)
sns.lineplot(usd_train["log"], color="limegreen", label="train log(Câmbio USD)")
sns.lineplot(usd_test["log"], color="magenta", label="test log(Câmbio USD)")
sns.lineplot(y=np.mean(usd_train["log"]), x=, color="black", linestyle="--", label="Média") # mean for usdo
sns.lineplot(y=np.max(usd_train["log"]), color="magenta", label="Máxima") # máxima for usdo
sns.lineplot(y=np.min(usd_train["log"]), color="magenta", linestyle="--", label="Mínima") # máxima for usdo
plt.title(f'Cotação do Euro - Série histórica ({usd_train.index[0].strftime(dt_format)} - {usd_test.index[-1].strftime(dt_format)})', fontsize="18")
plt.yticks(np.arange(round(df_wg_usd_6months["log"].min(), 2), round(df_wg_usd_6months["log"].max() + 0.01, 2), 0.01), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d, %b'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=20))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Câmbio log(USD ↔ BRL)", fontsize="18")
plt.legend(fontsize=18, loc="lower right", bbox_to_anchor=(0.99, 0.05, 0, 0))
plt.show()

# In[0.7]: Doing the seasonal decompose

usd_decom_multi = seasonal_decompose(usd_train["usd"], model="multiplicative")
usd_decom_multi.plot()

# Plotting season and residuals
plt.figure(figsize=(15,6))
usd_decom_multi.seasonal.plot(legend=True)
usd_decom_multi.resid.plot(legend=True)
plt.xlabel('Períodos',size=15)
plt.title('Sazonalidade e resíduos',size=15);

# plotting trend
plt.figure(figsize=(15,6))
usd_decom_multi.trend.plot(legend=True, color='r')
plt.xlabel('Períodos',size=15)
plt.title('Série de tendência',size=15);

# Plotting original data with trend:
sns.set_style("whitegrid",  {"grid.linestyle": ":"})
sns.set_palette("viridis")
plt.figure(figsize=(15, 10), dpi=300)
sns.lineplot(usd_train["usd"], color="limegreen", linewidth="2", label="Câmbio USD")
usd_decom_multi.trend.plot(color="darkblue", linewidth="2", label="Tendência")
plt.title(f'Cotação do Euro - Série histórica ({usd_train.index[0].strftime(dt_format)} - {usd_train.index[-1].strftime(dt_format)})', fontsize="18")
plt.yticks(np.arange(round(usd_train["usd"].min(), 1), round(usd_train["usd"].max() + 0.1, 1), 0.1), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d, %b'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=20))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Câmbio USD x Tendência", fontsize="18")
plt.legend(fontsize=18, loc="upper left", bbox_to_anchor=(0.02, 0, 0, 0.93))
plt.show()

print(f"Original series variance: {var_usd_3months}")
print(f"Trend variance: {np.var(usd_decom_multi.trend)/var_usd_3months}")
print(f"Seasonal variance: {np.var(usd_decom_multi.seasonal)/var_usd_3months}")
print(f"Residual Variance: {np.var(usd_decom_multi.resid)/var_usd_3months}")

kruskal(usd_train["usd"], usd_decom_multi.seasonal)

kruskal(usd_train["usd"], usd_decom_multi.trend)

kruskal(usd_train["usd"], usd_decom_multi.resid)

mannwhitneyu(usd_train["usd"], usd_decom_multi.seasonal)

mannwhitneyu(usd_train["usd"], usd_decom_multi.trend)

mannwhitneyu(usd_train["usd"], usd_decom_multi.resid)

f_oneway(usd_train["usd"], usd_decom_multi.seasonal)


usd_detrended = (usd_decom_multi.seasonal*usd_decom_multi.resid)
plt.figure(figsize=(15,6))
usd_detrended.plot()
plt.xlabel('Períodos',size=15)
plt.title('Gráfico da série detendencionalizada de forma multiplicativa',size=15);

usd_train['trend']=usd_decom_multi.trend
usd_train["sazon"] = usd_decom_multi.seasonal
usd_train['detrend']=usd_train['usd']/usd_train['trend']
usd_train["desazon"] = usd_train["usd"]/usd_train["sazon"]

#gráfico das séries original e detendenciada
plt.figure(figsize=(15,6), dpi=300)
sns.lineplot(usd_train["detrend"], color="darkblue", linewidth="2", label="Valores sem tendência")
sns.lineplot(usd_train["log"], color="darkorange", linewidth="2", label="Log dos valores reais")
plt.xlabel('Períodos',size=15)
plt.title('Série de consumo de cerveja com e sem tendência',size=15);
plt.legend(loc="upper left")

#gráfico das séries original e dessazonalizada
plt.figure(figsize=(15,6), dpi=300)
sns.lineplot(usd_train["desazon"], color="darkblue", linewidth="2", label="Valores dessazonalizados")
sns.lineplot(usd_train["usd"], color="darkorange", linewidth="2", label="Valores reais")
plt.xlabel('Períodos',size=15)
plt.title('Série de consumo de cerveja com e sem sazonalidade',size=15);
plt.legend(loc="upper left")


# In[1.0]: Defining Stationarity

# In[1.1]: Augmented Dickey-Fuller Test with usd and log

usd_3months_adf_ols = adfuller(usd_train['usd'], maxlag=None, autolag="AIC", store=True, regresults=True)[-1].resols.summary()
usd_3months_adf_ols

log_1year_adf = adfuller(usd_train["usd"], maxlag=None, autolag="AIC")
log_1year_adf

log_1year_kpss = kpss(usd_train["usd"], regression="c", nlags="auto")
log_1year_kpss

# Normal values results:
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.042
Model:                            OLS   Adj. R-squared:                  0.027
Method:                 Least Squares   F-statistic:                     2.761
Date:                Fri, 04 Oct 2024   Prob (F-statistic):             0.0671
Time:                        01:21:11   Log-Likelihood:                 226.42
No. Observations:                 128   AIC:                            -446.8
Df Residuals:                     125   BIC:                            -438.3
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1            -0.0322      0.017     -1.895      0.060      -0.066       0.001
x2             0.1296      0.088      1.477      0.142      -0.044       0.303
const          0.1756      0.091      1.927      0.056      -0.005       0.356
==============================================================================
Omnibus:                        1.214   Durbin-Watson:                   1.996
Prob(Omnibus):                  0.545   Jarque-Bera (JB):                0.807
Skew:                           0.162   Prob(JB):                        0.668
Kurtosis:                       3.214   Cond. No.                         138.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

# Doing the same for the log database

log_3months_adf_ols = adfuller(usd_train['log'], maxlag=None, autolag="AIC", store=True, regresults=True)[-1].resols.summary()
log_3months_adf_ols

# Results for log values:
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.043
Model:                            OLS   Adj. R-squared:                  0.027
Method:                 Least Squares   F-statistic:                     2.792
Date:                Fri, 04 Oct 2024   Prob (F-statistic):             0.0651
Time:                        01:22:12   Log-Likelihood:                 443.17
No. Observations:                 128   AIC:                            -880.3
Df Residuals:                     125   BIC:                            -871.8
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1            -0.0319      0.017     -1.915      0.058      -0.065       0.001
x2             0.1284      0.088      1.464      0.146      -0.045       0.302
const          0.0542      0.028      1.936      0.055      -0.001       0.110
==============================================================================
Omnibus:                        0.837   Durbin-Watson:                   1.994
Prob(Omnibus):                  0.658   Jarque-Bera (JB):                0.549
Skew:                           0.150   Prob(JB):                        0.760
Kurtosis:                       3.114   Cond. No.                         252.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

# We can notice that in the second model, both AIC and BIC are lower than the first one
# and also that the loglike is higher on the second model than the other, proving
# that the logarithmic version of the data is a better fit for the model so far.

# Getting values from the second model:

log_1year_adf = adfuller(usd_train["log"], maxlag=None, autolag="AIC")
log_1year_adf

# p-value is > 0.05 (0.32), DO NOT REJECT the null hypothesis and suggests the presence of a unit root (NON-STATIONARY)
# adf stats > 10% (-1.91), DO NOT REJECT the null hypothesis and suggests the presence of a unit root (NON-STATIONARY)
# lags = 3 good amount of lags

"""
(-1.9150540287021633,
 0.32501269667904653,
 1,
 128,
 {'1%': -3.4825006939887997,
  '5%': -2.884397984161377,
  '10%': -2.578960197753906},
 -790.7866605817836)
"""

# In[1.2]: Running KPSS test to determine stationarity for usd

log_1year_kpss = kpss(usd_train["log"], regression="c", nlags="auto")
log_1year_kpss

# p-value < 0.05 (0.01) REJECTS the null hypothesis, suggesting data is NON-STATIONARY
# kpss stat of 1.88 is > than 1% (0.739), REJECTS the null hypothesis, suggesting data is NON-STATIONARY
# lags = 10 good amount of lags

"""
(1.7077746917621264,
 0.01,
 6,
 {'10%': 0.347, '5%': 0.463, '2.5%': 0.574, '1%': 0.739})
"""

# In[1.3]: Plotting ACF and PACF to examine autocorrelations in data:

plt.style.use("seaborn-v0_8-colorblind")
fig, (ax1, ax2) = plt.subplots(2, figsize=(11, 7), dpi=600, sharex=True)
plot_acf(usd_train["log"], ax=ax1)
plot_pacf(usd_train["log"], ax=ax2)
plt.show()

# PACF plot shows an AR(2) order for the dataset showing a high statistical significance spike 
# at the first lag in PACF. ACF shows slow decay towards 0 -> exponentially decaying or sinusoidal

# In[1.4]: Defining the order of differencing we need:

# Original Series

plt.style.use("seaborn-v0_8-colorblind")
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(11, 8), dpi=600, sharex=True)
plt.xticks(fontsize="11")
ax1.spines['bottom'].set(linewidth=1.5, color="black")
ax1.spines['left'].set(linewidth=1.5, color="black")
ax1.plot(usd_train["log"])
ax1.set_title("Com log")
ax2.spines['bottom'].set(linewidth=1.5, color="black")
ax2.spines['left'].set(linewidth=1.5, color="black")
ax2.plot(usd_train["log"].diff().dropna())
ax2.set_title("Com Log e Diferenciação de 1ª ordem")
ax3.spines['bottom'].set(linewidth=1.5, color="black")
ax3.spines['left'].set(linewidth=1.5, color="black")
ax3.plot(usd_train["log"].diff().diff().dropna())
ax3.set_title("Com Log e Diferenciação de 2ª ordem")
# plt.suptitle("Comparação dos gráficos de Log com e sem diferenciação", fontsize="18")
plt.show()


# Plotting the ACF for each order

plt.style.use("seaborn-v0_8-colorblind")
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(11, 8), dpi=300, sharex=True)
plot_acf(usd_train["log"], ax=ax1)
ax1.spines['bottom'].set(linewidth=1.5, color="black")
ax1.spines['left'].set(linewidth=1.5, color="black")
ax1.set_title("Com Log")
plot_acf(usd_train["log"].diff().dropna(), ax=ax2)
ax2.spines['bottom'].set(linewidth=1.5, color="black")
ax2.spines['left'].set(linewidth=1.5, color="black")
ax2.set_title("Com Log e Diferenciação de 1ª ordem")
plot_acf(usd_train["log"].diff().diff().dropna(), ax=ax3)
ax3.spines['bottom'].set(linewidth=1.5, color="black")
ax3.spines['left'].set(linewidth=1.5, color="black")
ax3.set_title("Com Log e Diferenciação de 2ª ordem")
# plt.suptitle("Comparação da diferenciação na Autocorrelação (ACF)", fontsize="18")
plt.show()

# We can see a pretty good visually STATIONARY plot on the first differenciation,
# We can also see, after the first lag in the first diff ACF we have a pretty good
# white noise plot, and although we can see that the second lag goes into negative,
# meaning we could have a little overdifferentiation, we can deal with that in the
# final model by adding a number of MA terms at the final ARIMA. while in the
# second differentiation (second ACF) plot we can see that the second lag it goes
# straight and significantly into negative values, indicating a lot of over
# differentiation. I'll be going on with a single differenciation and I'll be
# looking for a number of MAs in the PACF plots next

# Plotting the PACf for the first order diff:

plt.style.use("seaborn-v0_8-colorblind")
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(11, 8), dpi=300, sharex=True)
plot_pacf(usd_train["log"], ax=ax1)
ax1.spines['bottom'].set(linewidth=1.5, color="black")
ax1.spines['left'].set(linewidth=1.5, color="black")
ax1.set_title("Com Log")
plot_pacf(usd_train["log"].diff().dropna(), ax=ax2)
ax2.spines['bottom'].set(linewidth=1.5, color="black")
ax2.spines['left'].set(linewidth=1.5, color="black")
ax2.set_title("Com Log e Diferenciação de 1ª ordem")
plot_pacf(usd_train["log"].diff().diff().dropna(), ax=ax3)
ax3.spines['bottom'].set(linewidth=1.5, color="black")
ax3.spines['left'].set(linewidth=1.5, color="black")
ax3.set_title("Com Log e Diferenciação de 2ª ordem")
# plt.suptitle("Comparação da diferenciação na Autocorrelação Parcial (PACF)", fontsize="18")
plt.show()

# Plotting ACF and PACF together:

fig, (ax1, ax2) = plt.subplots(2, figsize=(11,7), dpi=300)
ax1.spines['bottom'].set(linewidth=1.5, color="black")
ax1.spines['left'].set(linewidth=1.5, color="black")
ax2.spines['bottom'].set(linewidth=1.5, color="black")
ax2.spines['left'].set(linewidth=1.5, color="black")
plot_acf(usd_train["log"].diff().dropna(), ax=ax1)
plot_pacf(usd_train["log"].diff().dropna(), ax=ax2)
plt.show()

###############################################################################

# Plotting all ACF and PACF plots together
sns.set_palette("viridis")
sns.set_style("whitegrid",  {"grid.linestyle": ":"})
fig, axs = plt.subplots(3, 2, figsize=(11, 10), dpi=600, sharex=True)
(ax1, ax4, ax2, ax5, ax3, ax6) = axs.flat
plot_acf(usd_train["log"], ax=ax1)
ax1.set_title("ACF - Log", fontsize="14")
plot_acf(usd_train["log"].diff().dropna(), ax=ax2)
ax2.set_title("ACF - Log e Diferenciação de 1ª ordem", fontsize="14")
plot_acf(usd_train["log"].diff().diff().dropna(), ax=ax3)
ax3.set_title("ACF - Log e Diferenciação de 2ª ordem", fontsize="14")
plot_pacf(usd_train["log"], ax=ax4)
ax4.set_title("PACF - Log", fontsize="14")
plot_pacf(usd_train["log"].diff().dropna(), ax=ax5)
ax5.set_title("PACF - Log e Diferenciação de 1ª ordem", fontsize="14")
plot_pacf(usd_train["log"].diff().diff().dropna(), ax=ax6)
ax6.set_title("PACF - Com Log e Diferenciação de 2ª ordem", fontsize="14")
for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    ax.spines['bottom'].set(linewidth=1.5, color="black")
    ax.spines['left'].set(linewidth=1.5, color="black")
    ax.tick_params(axis="x", labelsize="12")
    ax.tick_params(axis="y", labelsize="12")
plt.savefig('./plots/save/acf_pacf_usd.tiff', dpi=600, format="tiff", bbox_inches="tight")
plt.show()

###############################################################################

# These plots show a sharp cut off at ACF lag 2, which indicates sligh overdifferencing
# and also an MA parameter of 2. The stationarized series display an "MA signature"
# Meaning we can explain the autocorrelation pattern my adding MA terms rather than
# AR terms. PACF is related to AR (orders while ACF is related to MA (lags of the forecast
# errors)

# These signatures need to be addressed after a series has been stationarized. So
# AR or MA signatures only should be addressed after stationarizing the series.
# This series showed an AR signature before being stationarized which implicated
# an AR(2) term, but after stationarizing, it now shows an MA(2) signature, which
# will indeed be added to the final model, while AR 

# AR(2)
# d = 1
# MA (2)

# In[1.5]: Plotting the differenced data for visual assessment:

fig, ax = plt.subplots(1, figsize=(15, 10))
ax.spines['bottom'].set(linewidth=1.5, color="black")
ax.spines['left'].set(linewidth=1.5, color="black")
sns.lineplot(usd_train["diff"], color="limegreen", label="Log(USD) - Diff")
plt.axhline(y=np.mean(usd_train["diff"]), color="black", linestyle="--", label="Média") # mean for usd
plt.title(f'Log do Euro Diferenciado - Série histórica ({usd_train.index[0].strftime(dt_format)} - {usd_train.index[-1].strftime(dt_format)})', fontsize="18")
plt.yticks(np.arange(round(usd_train["diff"].min(), 2), round(usd_train["diff"].max() + 0.01, 2), 0.01), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b, %Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=29))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Log(Câmbio USD ↔ BRL) Diferenciado ", fontsize="18")
plt.legend(fontsize=18, loc="lower left")
plt.show()

# In[2.0]: Defining Stationarity again

optimal_lags_usd_1year_diff = 12*(len(usd_train['diff'].dropna())/100)**(1/4)
optimal_lags_usd_1year_diff

# 12.813479668469292

# So optimal lags for this dataset = 12 or 13

# In[2.1]: Augmented Dickey-Fuller Test with diff

log_1year_diff_adf = adfuller(usd_train["diff"].dropna(), maxlag=None, autolag="AIC")
log_1year_diff_adf

# p-value is <<< 0.05 (5.988728689379452e-20), REJECTS the null hypothesis and suggests the absence of a unit root (STATIONARY)
# adf stats <<< 1% (-11), REJECTS the null hypothesis and suggests the absence of a unit root (STATIONARY)
# lags = 2 good amount of lags

"""
(-8.856542043563882,
 1.522509827535382e-14,
 1,
 128,
 {'1%': -3.4825006939887997,
  '5%': -2.884397984161377,
  '10%': -2.578960197753906},
 -411.66222602627056)
"""

log_1year_diff_adf_ols = adfuller(usd_train['diff'].dropna(), maxlag=None, autolag="AIC", store=True, regresults=True)[-1].resols.summary()
log_1year_diff_adf_ols

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.474
Model:                            OLS   Adj. R-squared:                  0.466
Method:                 Least Squares   F-statistic:                     56.32
Date:                Fri, 04 Oct 2024   Prob (F-statistic):           3.64e-18
Time:                        02:06:38   Log-Likelihood:                 234.29
No. Observations:                 128   AIC:                            -462.6
Df Residuals:                     125   BIC:                            -454.0
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1            -1.0683      0.121     -8.857      0.000      -1.307      -0.830
x2             0.1559      0.089      1.760      0.081      -0.019       0.331
const          0.0044      0.004      1.263      0.209      -0.003       0.011
==============================================================================
Omnibus:                        3.674   Durbin-Watson:                   1.974
Prob(Omnibus):                  0.159   Jarque-Bera (JB):                4.332
Skew:                          -0.021   Prob(JB):                        0.115
Kurtosis:                       3.900   Cond. No.                         39.9
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

# AIC, BIC and Loglike prove this model is not better than the second one, so
# I SHOULD USE d = 1 IN THE ARIMA MODEL and check d = 0

# In[2.2]: Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test post differencing

log_1year_diff_kpss = kpss(usd_train['diff'].diff().dropna(), regression="c", nlags="auto")
log_1year_diff_kpss

# p-value > 0.05 (0.1) DO NOT REJECT the null hypothesis, suggesting data is STATIONARY
# kpss stat of 0.06 is < 10% (0.347), DO NOT REJECT the null hypothesis, suggesting data is STATIONARY
# lags = 8 ok amount of lags for this point in time

"""
InterpolationWarning: The test statistic is outside of the range of p-values available in the
look-up table. The actual p-value is greater than the p-value returned.

(0.12368214525369713,
 0.1,
 24,
 {'10%': 0.347, '5%': 0.463, '2.5%': 0.574, '1%': 0.739})
"""

# KPSS suggests the need for a differencing of level 2. d = 2

# The adf test scares me a little bit as this could potentially mean I transformed the data
# into white noise, over-differencing it. In the other hand, the kpss test seems perfectly normal
# with a p-value far from 1, which could mean there is no over-differencing in this data.
# Understanding the number of lags that remained the same:
# The high number of lags here might indicate that there's still autocorrelation in the residuals
# with less lags. The lags in this tests are added to help the test adjust for residual autocorrelation
# and serial autocorrelation. It's not related to remainind autocorrelation in the dataset.

# In[2.3]: Plotting ACF and PACF to determine correct number of lags for usd

fig, (ax1, ax2) = plt.subplots(2, figsize=(11, 8), dpi=300)
plot_acf(usd_train["diff"].dropna(), ax=ax1)
plot_pacf(usd_train["diff"].dropna(), ax=ax2)
plt.show()

# Both tests seem to show normal stationary plots with no significant lags after zero.
# Also, no significant negative first lag;

# In[3.0]: Running Granger Causality tests

# In[3.1]: Create the the lagged dataframe:

usd_6months_lagged = pd.DataFrame({"diff": usd_train["diff"]})

usd_6months_lagged['lag 1'] = usd_6months_lagged["diff"].shift(1)
usd_6months_lagged['lag 2'] = usd_6months_lagged["diff"].shift(2)
usd_6months_lagged['lag 3'] = usd_6months_lagged["diff"].shift(3)
usd_6months_lagged['lag 4'] = usd_6months_lagged["diff"].shift(4)
usd_6months_lagged['lag 5'] = usd_6months_lagged["diff"].shift(5)
usd_6months_lagged['lag 6'] = usd_6months_lagged["diff"].shift(6)
usd_6months_lagged['lag 7'] = usd_6months_lagged["diff"].shift(7)
usd_6months_lagged['lag 8'] = usd_6months_lagged["diff"].shift(8)
usd_6months_lagged['lag 9'] = usd_6months_lagged["diff"].shift(9)
usd_6months_lagged['lag 10'] = usd_6months_lagged["diff"].shift(10)

usd_6months_lagged

# Running only one lag as this was what adf showed was optimal

# In[3.2]: Running Multicolinearity tests just in case:

usd_constants = add_constant(usd_6months_lagged.dropna())

usd_vif = pd.DataFrame()

usd_vif['vif'] = [variance_inflation_factor(usd_constants.values, i) for i in range(usd_constants.shape[1])]

usd_vif['variable'] = usd_constants.columns

usd_vif

"""
        vif variable
0  1.157827    const
1  1.069202     diff
2  1.082811    lag 1
3  1.095767    lag 2
4  1.068710    lag 3
5  1.089207    lag 4
6  1.112118    lag 5
7  1.105063    lag 6
8  1.095582    lag 7
"""

# We can see very high VIFs between the lags; Which means there's multicolinearity

# In[3.3]: Running the actual Causality test on the lagged data

usd_granger_lag1 = grangercausalitytests(usd_6months_lagged[["diff", "lag 5"]].dropna(), maxlag=5)

# No significant lags for this data

# In[3.4]: As an experiment, running the USD against USD:

usd_6months_granger = pd.read_csv("./datasets/arima_ready/eur_train_6months.csv", float_precision="high", parse_dates=([0]))

usd_6months_granger = pd.DataFrame(usd_6months_granger.drop("date", axis=1).values, index=pd.date_range(start="2024-03-25", periods=130 ,freq="B"), columns=["eur", "log", "diff"])

df_usd_usd = pd.DataFrame({"usd": usd_train['diff'], "eur": usd_6months_granger["diff"]})

usd_usd_granger = grangercausalitytests(df_usd_usd[['usd', 'eur']].dropna(), maxlag=30)

"""
Granger Causality
number of lags (no zero) 4
ssr based F test:         F=2.6640  , p=0.0359  , df_denom=117, df_num=4
ssr based chi2 test:   chi2=11.4759 , p=0.0217  , df=4
likelihood ratio test: chi2=10.9830 , p=0.0268  , df=4
parameter F test:         F=2.6640  , p=0.0359  , df_denom=117, df_num=4

Granger Causality
number of lags (no zero) 5
ssr based F test:         F=2.5074  , p=0.0341  , df_denom=114, df_num=5
ssr based chi2 test:   chi2=13.7465 , p=0.0173  , df=5
likelihood ratio test: chi2=13.0418 , p=0.0230  , df=5
parameter F test:         F=2.5074  , p=0.0341  , df_denom=114, df_num=5

Granger Causality
number of lags (no zero) 6
ssr based F test:         F=2.1950  , p=0.0486  , df_denom=111, df_num=6
ssr based chi2 test:   chi2=14.7127 , p=0.0226  , df=6
likelihood ratio test: chi2=13.9033 , p=0.0307  , df=6
parameter F test:         F=2.1950  , p=0.0486  , df_denom=111, df_num=6
"""

# Significant p-value on lags 4, 5 and 6.

# In[3.6]: Cross-testing the series to make sure they have causality between them

tscv = TimeSeriesSplit(n_splits=7)

ct_usd_usd = df_usd_usd.dropna()

for train_index, test_index in tscv.split(ct_usd_usd):
    train, test = ct_usd_usd.iloc[train_index], ct_usd_usd.iloc[test_index]
    
    X_train, y_train = train['eur'], train['usd']
    X_test, y_test = test['eur'], test['usd']
    
    granger_result = grangercausalitytests(train[['usd', 'eur']], maxlag=4, verbose=False)
    
    for lag, result in granger_result.items():
        f_test_pvalue = result[0]['ssr_ftest'][1]  # p-value from F-test
        print(f"Lag: {lag}, P-value: {f_test_pvalue}")
    
    print(f"TRAIN indices: {train_index}")
    print(f"TEST indices: {test_index}")


"""
Lag: 1, P-value: 0.5768887777876386
Lag: 2, P-value: 0.9236327585934487
Lag: 3, P-value: 0.16196780582800277
Lag: 4, P-value: 0.010457902442415911

Lag: 1, P-value: 0.14555642952426098
Lag: 2, P-value: 0.45205955832043
Lag: 3, P-value: 0.04989799237837883
Lag: 4, P-value: 0.010928439023283049

Lag: 1, P-value: 0.18820089863757525
Lag: 2, P-value: 0.5902284410457266
Lag: 3, P-value: 0.2079536877957213
Lag: 4, P-value: 0.03640727743426027

Lag: 1, P-value: 0.4344305998655742
Lag: 2, P-value: 0.8285452154161301
Lag: 3, P-value: 0.06571138355133598
Lag: 4, P-value: 0.006189136618057291

Lag: 1, P-value: 0.6132502760026286
Lag: 2, P-value: 0.6779396937563593
Lag: 3, P-value: 0.02797476768277951
Lag: 4, P-value: 0.005665738028894668
"""

# There is significant causality in lag 4

# In[4.0]: Save final dataset for testing ARIMA

usd_train_6months = pd.DataFrame({
    "date": usd_train.index,
    "usd": usd_train["usd"],
    "log": usd_train["log"],
    "diff": usd_train["diff"],
    "eur": usd_6months_granger["diff"]
    })

usd_test_6months = pd.DataFrame({
    "date": usd_test.index,
    "usd": usd_test["usd"],
    "log": usd_test["log"],
    "diff": usd_test["diff"]
    })

# save to csv
usd_train_6months.to_csv("./datasets/arima_ready/usd_train_6months.csv", index=False)
usd_test_6months.to_csv("./datasets/arima_ready/usd_test_6months.csv", index=False)




