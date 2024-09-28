# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # #
# Exploratory Data Analysis = EURO - 365 days #
# # # # # # # # # # # # # # # # # # # # # # # #

# In[0.1]: Importação dos pacotes

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import boxcox, norm
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime


# In[0.2]: Import dataframes

df_wg_eur_1year = pd.read_csv("../datasets/wrangled/df_eur_1year.csv", float_precision="high", parse_dates=([1]))

df_wg_eur_1year.info()

df_wg_eur_5year = pd.read_csv("../datasets/wrangled/df_eur_5year.csv", float_precision="high", parse_dates=([1]))

df_wg_eur_5year

dt_format = "%d/%m/%Y"

df_wg_eur_1year = df_wg_eur_1year.drop(df_wg_eur_1year[df_wg_eur_1year["weekday"] == "Sunday"].index)

# In[0.3]: Separate data into train and test

# remove last 15 days of data for test later
eur_train = df_wg_eur_1year[:-15]

eur_test = df_wg_eur_1year[-15:]

# In[0.4]: Plot original datasets

# 5 year dataset
sns.set(palette="viridis")
sns.set_style("whitegrid",  {"grid.linestyle": ":"})
plt.figure(figsize=(15, 10), dpi=300)
sns.lineplot(x=df_wg_eur_5year["dateTime"], y=df_wg_eur_5year["eur"], color="limegreen", label="Câmbio EUR")
plt.axhline(y=np.mean(df_wg_eur_5year["eur"]), color="black", linestyle="--", label="Média", linewidth=2) # mean for euro
plt.axhline(y=np.max(df_wg_eur_5year["eur"]), color="magenta", label="Máxima", linewidth=2) # máxima for euro
plt.axhline(y=np.min(df_wg_eur_5year["eur"]), color="magenta", linestyle="--", label="Mínima", linewidth=2) # máxima for euro
plt.title(f'Cotação do Euro - Série histórica ({df_wg_eur_5year["dateTime"][0].strftime(dt_format)} - {df_wg_eur_5year["dateTime"][1825].strftime(dt_format)})', fontsize="18")
plt.yticks(np.arange(round(df_wg_eur_5year["eur"].min(), 1), round(df_wg_eur_5year["eur"].max() + 0.1, 1), 0.2), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b, %Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=180))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Câmbio EUR ↔ BRL", fontsize="18")
plt.legend(fontsize=16, loc="upper right", bbox_to_anchor=(0.98, 0, 0, 0.93))
plt.show()

# 1 year dataset
sns.set_style("whitegrid",  {"grid.linestyle": ":"})
plt.figure(figsize=(15, 10), dpi=300)
sns.lineplot(x=df_wg_eur_1year["dateTime"], y=df_wg_eur_1year["eur"], color="limegreen", label="Câmbio EUR")
plt.axhline(y=np.mean(df_wg_eur_1year["eur"]), color="black", linestyle="--", label="Média", linewidth=2) # mean for euro
plt.axhline(y=np.max(df_wg_eur_1year["eur"]), color="magenta", label="Máxima", linewidth=2) # máxima for euro
plt.axhline(y=np.min(df_wg_eur_1year["eur"]), color="magenta", linestyle="--", label="Mínima", linewidth=2) # máxima for euro
plt.title(f'Cotação do Euro - Série histórica ({df_wg_eur_1year["dateTime"][0].strftime(dt_format)} - {df_wg_eur_1year["dateTime"][279].strftime(dt_format)})', fontsize="18")
plt.yticks(np.arange(round(df_wg_eur_1year["eur"].min(), 1), round(df_wg_eur_1year["eur"].max() + 0.1, 1), 0.1), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b, %Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=29))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Câmbio EUR ↔ BRL", fontsize="18")
plt.legend(fontsize=18, loc="upper left", bbox_to_anchor=(0.02, 0, 0, 0.93))
plt.show()

# Boxplot for 5 years:

plt.figure(figsize=(15,10), dpi=300)
sns.boxplot(x=df_wg_eur_5year["eur"], palette="viridis")
plt.title("Boxplot do Dataset - 5 anos (1826 dias)", fontsize="18")
plt.show()

# Boxplot for 1 year:

plt.figure(figsize=(15,10), dpi=300)
sns.boxplot(x=df_wg_eur_1year["eur"], palette="viridis")
plt.title("Boxplot do Dataset - 1 ano (365 dias)", fontsize="18")
plt.legend(fontsize="16")
plt.show()


# In[0.5]: Plot the train data:

sns.set_style("whitegrid",  {"grid.linestyle": ":"})
plt.figure(figsize=(15, 10), dpi=300)
sns.lineplot(x=eur_train["dateTime"], y=eur_train["eur"], color="limegreen", label="Câmbio EUR")
plt.axhline(y=np.mean(eur_train["eur"]), color="black", linestyle="--", label="Média", linewidth=2) # mean for euro
plt.axhline(y=np.max(eur_train["eur"]), color="magenta", label="Máxima", linewidth=2) # máxima for euro
plt.axhline(y=np.min(eur_train["eur"]), color="magenta", linestyle="--", label="Mínima", linewidth=2) # máxima for euro
plt.title(f'Cotação do Euro - Treino - Série histórica ({eur_train["dateTime"][0].strftime(dt_format)} - {eur_train["dateTime"][279].strftime(dt_format)})', fontsize="18")
plt.yticks(np.arange(round(eur_train["eur"].min(), 1), round(eur_train["eur"].max() + 0.1, 1), 0.1), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b, %Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=29))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Câmbio EUR ↔ BRL", fontsize="18")
plt.legend(fontsize=18, loc="upper left", bbox_to_anchor=(0.02, 0, 0, 0.93))
plt.show()

plt.figure(figsize=(15,10), dpi=300)
sns.boxplot(x=eur_train["eur"], palette="viridis")
plt.title("Boxplot do Dataset - 1 ano (365 dias)", fontsize="18")
plt.legend(fontsize="16")
plt.show()

plt.figure(figsize=(15,10), dpi=300)
sns.boxplot(x=eur_train["log"], palette="viridis")
plt.title("Boxplot do Dataset - 1 ano (365 dias)", fontsize="18")
plt.legend(fontsize="16")
plt.show()

plt.figure(figsize=(15,10), dpi=300)
sns.boxplot(x=eur_train["diff"], palette="viridis")
plt.title("Boxplot do Dataset - 1 ano (365 dias)", fontsize="18")
plt.legend(fontsize="16")
plt.show()



# In[0.4]: Calculate Statistics for datasets

var_eur_1year = np.var(eur_train['eur'])
var_eur_1year

# 0.08255265745980815

varlog_eur_1year = np.var(eur_train['log'])
varlog_eur_1year

# 0.0025448361292852145

optimal_lags_eur_1year = 12*(len(df_wg_eur_1year['eur'])/100)**(1/4)
optimal_lags_eur_1year

# 15.961265820308224

# So optimal lags for this dataset = 15 ou 16

# In[0.7]: Plot train and test datasets:

plt.figure(figsize=(15, 10), dpi=300)
sns.lineplot(x=eur_train["dateTime"], y=eur_train["log"], color="limegreen", label="log(Câmbio EUR)")
sns.lineplot(x=eur_test["dateTime"], y=eur_test["log"], color="magenta", label="test log(Câmbio EUR)")
plt.axhline(y=np.mean(eur_train["log"]), color="black", linestyle="--", label="Média") # mean for euro
plt.axhline(y=np.max(eur_train["log"]), color="magenta", label="Máxima") # máxima for euro
plt.axhline(y=np.min(eur_train["log"]), color="magenta", linestyle="--", label="Mínima") # máxima for euro
plt.title(f'Cotação do Euro - Série histórica ({eur_train["dateTime"][0].strftime(dt_format)} - {eur_test["dateTime"][364].strftime(dt_format)})', fontsize="18")
plt.yticks(np.arange(round(df_wg_eur_1year["log"].min(), 2), round(df_wg_eur_1year["log"].max() + 0.01, 2), 0.01), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b, %Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=29))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Câmbio log(EUR ↔ BRL)", fontsize="18")
plt.legend(fontsize=18, loc="upper left", bbox_to_anchor=(0.02, 0, 0, 0.93))
plt.show()

# Plot "closer"
plt.figure(figsize=(15, 10), dpi=300)
sns.lineplot(x=eur_train["dateTime"][200:], y=eur_train["log"][200:], color="limegreen", label="log(Câmbio EUR)")
sns.lineplot(x=eur_test["dateTime"], y=eur_test["log"], color="magenta", label="test log(Câmbio EUR)")
plt.axhline(y=np.max(eur_train["log"]), color="magenta", label="Máxima") # máxima for euro
plt.title(f'Cotação do Euro - Série histórica ({eur_train["dateTime"][200].strftime(dt_format)} - {eur_test["dateTime"][364].strftime(dt_format)})', fontsize="18")
plt.yticks(np.arange(round(df_wg_eur_1year["log"][200:].min(), 2), round(df_wg_eur_1year["log"][200:].max() + 0.01, 2), 0.01), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b, %Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=15))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Câmbio log(EUR ↔ BRL)", fontsize="18")
plt.legend(fontsize=18, loc="lower right")
plt.show()

# In[0.7]: Plot the log dataset for visual assessment

plt.figure(figsize=(15, 10), dpi=300)
sns.lineplot(x=df_wg_eur_1year["dateTime"], y=df_wg_eur_1year["log"], color="limegreen", label="log(Câmbio EUR)")
plt.axhline(y=np.mean(df_wg_eur_1year["log"]), color="black", linestyle="--", label="Média") # mean for euro
plt.axhline(y=np.max(df_wg_eur_1year["log"]), color="magenta", label="Máxima") # máxima for euro
plt.axhline(y=np.min(df_wg_eur_1year["log"]), color="magenta", linestyle="--", label="Mínima") # máxima for euro
plt.title(f'Cotação do Euro - Série histórica ({df_wg_eur_1year["dateTime"][0].strftime(dt_format)} - {df_wg_eur_1year["dateTime"][279].strftime(dt_format)})', fontsize="18")
plt.yticks(np.arange(round(df_wg_eur_1year["log"].min(), 2), round(df_wg_eur_1year["log"].max() + 0.01, 2), 0.01), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b, %Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=29))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Câmbio log(EUR ↔ BRL)", fontsize="18")
plt.legend(fontsize=18, loc="upper left", bbox_to_anchor=(0.02, 0, 0, 0.93))
plt.show()


# In[1.0]: Defining Stationarity

# In[1.1]: Augmented Dickey-Fuller Test with eur and log

eur_1year_adf_ols = adfuller(eur_train['eur'], maxlag=None, autolag="AIC", store=True, regresults=True)[-1].resols.summary()
eur_1year_adf_ols

# Normal values results:
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.028
Model:                            OLS   Adj. R-squared:                  0.015
Method:                 Least Squares   F-statistic:                     2.092
Date:                Fri, 27 Sep 2024   Prob (F-statistic):             0.0819
Time:                        23:45:00   Log-Likelihood:                 584.34
No. Observations:                 294   AIC:                            -1159.
Df Residuals:                     289   BIC:                            -1140.
Df Model:                           4                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1             0.0028      0.007      0.403      0.687      -0.011       0.017
x2            -0.0190      0.058     -0.326      0.745      -0.134       0.096
x3             0.0288      0.058      0.493      0.623      -0.086       0.144
x4            -0.1645      0.059     -2.800      0.005      -0.280      -0.049
const         -0.0119      0.039     -0.306      0.760      -0.088       0.065
==============================================================================
Omnibus:                       11.838   Durbin-Watson:                   1.996
Prob(Omnibus):                  0.003   Jarque-Bera (JB):               25.142
Skew:                           0.074   Prob(JB):                     3.47e-06
Kurtosis:                       4.425   Cond. No.                         176.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

# Doing the same for the log database

log_1year_adf_ols = adfuller(eur_train['log'], maxlag=None, autolag="AIC", store=True, regresults=True)[-1].resols.summary()
log_1year_adf_ols

# Results for log values:
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.024
Model:                            OLS   Adj. R-squared:                  0.011
Method:                 Least Squares   F-statistic:                     1.809
Date:                Fri, 27 Sep 2024   Prob (F-statistic):              0.127
Time:                        23:45:30   Log-Likelihood:                 1094.7
No. Observations:                 294   AIC:                            -2179.
Df Residuals:                     289   BIC:                            -2161.
Df Model:                           4                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1             0.0024      0.007      0.344      0.731      -0.011       0.016
x2            -0.0283      0.059     -0.483      0.629      -0.144       0.087
x3             0.0101      0.059      0.172      0.863      -0.105       0.125
x4            -0.1544      0.059     -2.627      0.009      -0.270      -0.039
const         -0.0035      0.012     -0.289      0.773      -0.027       0.020
==============================================================================
Omnibus:                       10.221   Durbin-Watson:                   1.996
Prob(Omnibus):                  0.006   Jarque-Bera (JB):               19.286
Skew:                           0.098   Prob(JB):                     6.49e-05
Kurtosis:                       4.239   Cond. No.                         349.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

# We can notice that in the second model, both AIC and BIC are lower than the first one
# and also that the loglike is higher on the second model than the other, proving
# that the logarithmic version of the data is a better fit for the model so far.

# Getting values from the second model:

log_1year_adf = adfuller(eur_train["log"], maxlag=None, autolag="AIC")
log_1year_adf

# p-value is > 0.05 (0.97), DO NOT REJECT the null hypothesis and suggests the presence of a unit root (NON-STATIONARY)
# adf stats > 10% (0.34), DO NOT REJECT the null hypothesis and suggests the presence of a unit root (NON-STATIONARY)
# lags = 3 good amount of lags

"""
(0.3439623687915441,
 0.9792739515169508,
 3,
 294,
 {'1%': -3.452789844280995,
  '5%': -2.871421512222641,
  '10%': -2.5720351510944512},
 -2088.856318151855)
"""

# In[1.2]: Running KPSS test to determine stationarity for eur

log_1year_kpss = kpss(eur_train["log"], regression="c", nlags="auto")
log_1year_kpss

# p-value < 0.05 (0.01) REJECTS the null hypothesis, suggesting data is NON-STATIONARY
# kpss stat of 2.35 is > than 1% (0.739), REJECTS the null hypothesis, suggesting data is NON-STATIONARY
# lags = 10 good amount of lags

"""
(2.3573104119194657,
 0.01,
 10,
 {'10%': 0.347, '5%': 0.463, '2.5%': 0.574, '1%': 0.739})
"""

# In[1.3]: Plotting ACF and PACF to examine autocorrelations in data:

plt.style.use("seaborn-v0_8-colorblind")
fig, (ax1, ax2) = plt.subplots(2, figsize=(11, 7), dpi=300, sharex=True)
plot_acf(eur_train["log"], ax=ax1)
plot_pacf(eur_train["log"], ax=ax2)
plt.show()

# PACF plot shows an AR(2) order for the dataset showing a high statistical significance spike 
# at the first lag in PACF. ACF shows slow decay towards 0 -> exponentially decaying or sinusoidal

# In[1.4]: Defining the order of differencing we need:

# Original Series

plt.style.use("seaborn-v0_8-colorblind")
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(11, 8), dpi=300, sharex=True)
ax1.plot(eur_train["log"])
ax1.set_title("Com log")
ax2.plot(eur_train["diff"].dropna(), color="green")
ax2.set_title("Com Log e Diferenciação de 1ª ordem")
ax3.plot(eur_train["diff"].diff().dropna())
ax3.set_title("Com Log e Diferenciação de 2ª ordem")
plt.suptitle("Comparação dos gráficos de Log com e sem diferenciação", fontsize="18")
plt.show()


# Plotting the ACF for each order

plt.style.use("seaborn-v0_8-colorblind")
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(11, 8), dpi=300, sharex=True)
plot_acf(eur_train["log"], ax=ax1)
ax1.set_title("Com Log")
plot_acf(eur_train["diff"].dropna(), ax=ax2, color="green")
ax2.set_title("Com Log e Diferenciação de 1ª ordem")
plot_acf(eur_train["diff"].diff().dropna(), ax=ax3)
ax3.set_title("Com Log e Diferenciação de 2ª ordem")
plt.suptitle("Comparação da diferenciação na Autocorrelação (ACF)", fontsize="18")
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
plot_pacf(eur_train["log"], ax=ax1)
ax1.set_title("Com Log")
plot_pacf(eur_train["diff"].dropna(), ax=ax2, color="green")
ax2.set_title("Com Log e Diferenciação de 1ª ordem")
plot_pacf(eur_train["diff"].diff().dropna(), ax=ax3)
ax3.set_title("Com Log e Diferenciação de 2ª ordem")
plt.suptitle("Comparação da diferenciação na Autocorrelação Parcial (PACF)", fontsize="18")
plt.show()

# Plotting ACF and PACF together:

fig, (ax1, ax2) = plt.subplots(2, figsize=(11,7), dpi=300)
plot_acf(eur_train["diff"].dropna(), ax=ax1)
plot_pacf(eur_train["diff"].dropna(), ax=ax2)
plt.show()

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

# In[1.5]: Plotting the differenced data for visual assessment:

plt.figure(figsize=(15, 10))
sns.lineplot(x=eur_train["dateTime"], y=eur_train["diff"], color="limegreen", label="Log(EUR) - Diff")
plt.axhline(y=np.mean(eur_train["diff"]), color="black", linestyle="--", label="Média") # mean for eur
plt.title(f'Log do Euro Diferenciado - Série histórica ({eur_train["dateTime"][0].strftime(dt_format)} - {eur_train["dateTime"][297].strftime(dt_format)})', fontsize="18")
plt.yticks(np.arange(round(eur_train["diff"].min(), 2), round(eur_train["diff"].max() + 0.01, 2), 0.01), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b, %Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=29))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Log(Câmbio EUR ↔ BRL) Diferenciado ", fontsize="18")
plt.legend(fontsize=18, loc="upper center")
plt.show()

# In[2.0]: Defining Stationarity again

optimal_lags_eur_1year_diff = 12*(len(eur_train['diff'].dropna())/100)**(1/4)
optimal_lags_eur_1year_diff

# 14.920408761353036

# So optimal lags for this dataset = 14 or 15

# In[2.1]: Augmented Dickey-Fuller Test with diff

log_1year_diff_adf = adfuller(eur_train["diff"].dropna(), maxlag=None, autolag="AIC")
log_1year_diff_adf

# p-value is <<< 0.05 (5.988728689379452e-20), REJECTS the null hypothesis and suggests the absence of a unit root (STATIONARY)
# adf stats <<< 1% (-11), REJECTS the null hypothesis and suggests the absence of a unit root (STATIONARY)
# lags = 2 good amount of lags

"""
(-11.374716739805612,
 8.803103734769128e-21,
 2,
 294,
 {'1%': -3.452789844280995,
  '5%': -2.871421512222641,
  '10%': -2.5720351510944512},
 -1107.8164425188288)
"""

log_1year_diff_adf_ols = adfuller(eur_train['diff'].dropna(), maxlag=None, autolag="AIC", store=True, regresults=True)[-1].resols.summary()
log_1year_diff_adf_ols

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.525
Model:                            OLS   Adj. R-squared:                  0.520
Method:                 Least Squares   F-statistic:                     106.7
Date:                Sat, 28 Sep 2024   Prob (F-statistic):           1.46e-46
Time:                        00:39:57   Log-Likelihood:                 584.26
No. Observations:                 294   AIC:                            -1161.
Df Residuals:                     290   BIC:                            -1146.
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1            -1.1460      0.101    -11.375      0.000      -1.344      -0.948
x2             0.1297      0.083      1.563      0.119      -0.034       0.293
x3             0.1614      0.058      2.775      0.006       0.047       0.276
const          0.0037      0.002      1.885      0.060      -0.000       0.008
==============================================================================
Omnibus:                       11.903   Durbin-Watson:                   1.995
Prob(Omnibus):                  0.003   Jarque-Bera (JB):               24.577
Skew:                           0.108   Prob(JB):                     4.60e-06
Kurtosis:                       4.400   Cond. No.                         67.3
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

# AIC, BIC and Loglike prove this model is even better than the second one, so
# I SHOULD USE D = 1 IN THE ARIMA MODEL

# In[2.2]: Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test post differencing

log_1year_diff_kpss = kpss(eur_train['diff'].dropna(), regression="c", nlags="auto")
log_1year_diff_kpss

# p-value > 0.05 (0.1) DO NOT REJECT the null hypothesis, suggesting data is STATIONARY
# kpss stat of 0.22 is < 10% (0.347), DO NOT REJECT the null hypothesis, suggesting data is STATIONARY
# lags = 8 ok amount of lags for this point in time

"""
(0.2279711249901558,
 0.1,
 8,
 {'10%': 0.347, '5%': 0.463, '2.5%': 0.574, '1%': 0.739})
"""

# The adf test scares me a little bit as this could potentially mean I transformed the data
# into white noise, over-differencing it. In the other hand, the kpss test seems perfectly normal
# with a p-value far from 1, which could mean there is no over-differencing in this data.
# Understanding the number of lags that remained the same:
# The high number of lags here might indicate that there's still autocorrelation in the residuals
# with less lags. The lags in this tests are added to help the test adjust for residual autocorrelation
# and serial autocorrelation. It's not related to remainind autocorrelation in the dataset.

# In[2.3]: Plotting ACF and PACF to determine correct number of lags for eur

fig, (ax1, ax2) = plt.subplots(2, figsize=(11, 8), dpi=300)
plot_acf(eur_train["diff"].dropna(), ax=ax1)
plot_pacf(eur_train["diff"].dropna(), ax=ax2)
plt.show()

# Both tests seem to show normal stationary plots with no significant lags after zero.
# Also, no significant negative first lag;

# In[3.0]: Running Granger Causality tests

# In[3.1]: Create the the lagged dataframe:

eur_1year_lagged = pd.DataFrame({"log": eur_train["log"]})

eur_1year_lagged['lag 1'] = eur_1year_lagged["log"].shift(1)
eur_1year_lagged['lag 2'] = eur_1year_lagged["log"].shift(2)
eur_1year_lagged['lag 3'] = eur_1year_lagged["log"].shift(3)
eur_1year_lagged['lag 4'] = eur_1year_lagged["log"].shift(4)
eur_1year_lagged['lag 5'] = eur_1year_lagged["log"].shift(5)
eur_1year_lagged['lag 6'] = eur_1year_lagged["log"].shift(6)
eur_1year_lagged['lag 7'] = eur_1year_lagged["log"].shift(7)

eur_1year_lagged

# Running only one lag as this was what adf showed was optimal

# In[3.2]: Running Multicolinearity tests just in case:

eur_constants = add_constant(eur_1year_lagged.dropna())

eur_vif = pd.DataFrame()

eur_vif['vif'] = [variance_inflation_factor(eur_constants.values, i) for i in range(eur_constants.shape[1])]

eur_vif['variable'] = eur_constants.columns

eur_vif

"""
           vif variable
0  1271.623916    const
1    75.021847      log
2   142.571719    lag 1
3   140.284288    lag 2
4   140.793289    lag 3
5   140.132377    lag 4
6   136.024549    lag 5
7   134.689439    lag 6
8    69.642098    lag 7
"""

# We can see very high VIFs between the lags; Which means there's multicolinearity

# In[3.3]: Running the actual Causality test on the lagged data

eur_granger_lag2 = grangercausalitytests(eur_1year_lagged[["log", "lag 2"]].dropna(), maxlag=4)

"""
Granger Causality
number of lags (no zero) 2
ssr based F test:         F=3.4914  , p=0.0318  , df_denom=289, df_num=2
ssr based chi2 test:   chi2=7.1035  , p=0.0287  , df=2
likelihood ratio test: chi2=7.0191  , p=0.0299  , df=2
parameter F test:         F=3.4914  , p=0.0318  , df_denom=289, df_num=2
"""

eur_granger_lag6 = grangercausalitytests(eur_1year_lagged[["log", "lag 6"]].dropna(), maxlag=4)

"""
Granger Causality
number of lags (no zero) 1
ssr based F test:         F=4.4268  , p=0.0362  , df_denom=288, df_num=1
ssr based chi2 test:   chi2=4.4730  , p=0.0344  , df=1
likelihood ratio test: chi2=4.4389  , p=0.0351  , df=1
parameter F test:         F=4.4268  , p=0.0362  , df_denom=288, df_num=1
"""

eur_granger_lag7 = grangercausalitytests(eur_1year_lagged[["log", "lag 7"]].dropna(), maxlag=4)

"""
Granger Causality
number of lags (no zero) 1
ssr based F test:         F=5.7418  , p=0.0172  , df_denom=287, df_num=1
ssr based chi2 test:   chi2=5.8018  , p=0.0160  , df=1
likelihood ratio test: chi2=5.7445  , p=0.0165  , df=1
parameter F test:         F=5.7418  , p=0.0172  , df_denom=287, df_num=1

Granger Causality
number of lags (no zero) 3
ssr based F test:         F=3.0852  , p=0.0277  , df_denom=281, df_num=3
ssr based chi2 test:   chi2=9.4862  , p=0.0235  , df=3
likelihood ratio test: chi2=9.3334  , p=0.0252  , df=3
parameter F test:         F=3.0852  , p=0.0277  , df_denom=281, df_num=3
"""

# Choosing "lag 7" as it shows better p-values at lags 1 and 3

# I've run other amount of lags for all of these, but mostly have p-values close
# to 1.

# In[3.4]: As an experiment, running the EUR against USD:

usd_1year_granger = pd.read_csv("../datasets/wrangled/df_usd_1year.csv", float_precision="high", parse_dates=([0]))

df_eur_usd = pd.DataFrame({"eur": eur_train['log'], "usd": usd_1year_granger["log"]})

eur_usd_granger = grangercausalitytests(df_eur_usd[['eur', 'usd']].dropna(), maxlag=40)

# None of the lags show any statistical p-values for any of the tests.

# In[3.6]: Cross-testing the series to make sure they have causality between them

tscv = TimeSeriesSplit(n_splits=5)

ct_eur_usd = df_eur_usd.dropna()

for train_index, test_index in tscv.split(ct_eur_usd):
    train, test = ct_eur_usd.iloc[train_index], ct_eur_usd.iloc[test_index]
    
    X_train, y_train = train['usd'], train['eur']
    X_test, y_test = test['usd'], test['eur']
    
    granger_result = grangercausalitytests(train[['eur', 'usd']], maxlag=4, verbose=False)
    
    for lag, result in granger_result.items():
        f_test_pvalue = result[0]['ssr_ftest'][1]  # p-value from F-test
        print(f"Lag: {lag}, P-value: {f_test_pvalue}")
    
    print(f"TRAIN indices: {train_index}")
    print(f"TEST indices: {test_index}")


# No significant causality between the two datasets

# In[4.0]: Save final dataset for testing ARIMA

eur_train_1year = pd.DataFrame({
    "date": eur_train["dateTime"],
    "eur": eur_train["eur"],
    "log": eur_train["log"],
    "diff": eur_train["diff"],
    "lag7": eur_1year_lagged["lag 7"]
    })

eur_test_1year = eur_test

# save to csv
eur_train_1year.to_csv("../datasets/arima_ready/eur_train_1year.csv", index=False)
eur_test_1year.to_csv("../datasets/arima_ready/eur_test_1year.csv", index=False)
