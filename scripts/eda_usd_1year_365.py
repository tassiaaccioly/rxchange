# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # #
# Exploratory Data Analysis = USDO - 1 year #
# # # # # # # # # # # # # # # # # # # # # # #

# In[0.1]: Importação dos pacotes

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.model_selection import TimeSeriesSplit


# In[0.2]: Import dataframes

df_wg_usd_1year = pd.read_csv("./datasets/wrangled/df_usd_1year.csv", float_precision="high", parse_dates=([1]))

df_wg_usd_1year.info()

df_wg_usd_5year = pd.read_csv("./datasets/wrangled/df_usd_5year.csv", float_precision="high", parse_dates=([1]))

df_wg_usd_5year

dt_format = "%d/%m/%Y"

# In[0.3]: Separate data into train and test

# remove last 15 days of data for test later
usd_train = df_wg_usd_1year[:-15]

# keep only the last 15 days of data
usd_test = df_wg_usd_1year[-15:]

# In[0.3]: Plot the original series

# 5 years

sns.set(palette="viridis")
sns.set_style("whitegrid",  {"grid.linestyle": ":"})
plt.figure(figsize=(15, 10), dpi=300)
sns.lineplot(x=df_wg_usd_5year["dateTime"], y=df_wg_usd_5year["usd"], color="limegreen", label="Câmbio USD")
plt.axhline(y=np.mean(df_wg_usd_5year["usd"]), color="black", linestyle="--", label="Média", linewidth=2) # mean for usd
plt.axhline(y=np.max(df_wg_usd_5year["usd"]), color="magenta", label="Máxima", linewidth=2) # max for usd
plt.axhline(y=np.min(df_wg_usd_5year["usd"]), color="magenta", linestyle="--", label="Mínima", linewidth=2) # min for usd
plt.title(f'Cotação do Dóllar - Série histórica ({df_wg_usd_5year["dateTime"][0].strftime(dt_format)} - {df_wg_usd_5year["dateTime"][1825].strftime(dt_format)})', fontsize="18")
plt.yticks(np.arange(round(df_wg_usd_5year["usd"].min(), 1), round(df_wg_usd_5year["usd"].max() + 0.1, 1), 0.2), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b, %Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=180))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Câmbio USD ↔ BRL", fontsize="18")
plt.legend(fontsize=18, loc="upper right", bbox_to_anchor=(0.98, 0, 0, 0.32))
plt.show()

# 1 year

plt.figure(figsize=(15, 10), dpi=300)
sns.lineplot(x=df_wg_usd_1year["dateTime"], y=df_wg_usd_1year["usd"], color="limegreen", label="Câmbio USD")
plt.axhline(y=np.mean(df_wg_usd_1year["usd"]), color="black", linestyle="--", label="Média", linewidth=2) # mean for usd
plt.axhline(y=np.max(df_wg_usd_1year["usd"]), color="magenta", label="Máxima", linewidth=2) # max for usd
plt.axhline(y=np.min(df_wg_usd_1year["usd"]), color="magenta", linestyle="--", label="Mínima", linewidth=2) # min for usd
plt.title(f'Cotação do Euro - Série histórica ({df_wg_usd_1year["dateTime"][0].strftime(dt_format)} - {df_wg_usd_1year["dateTime"][312].strftime(dt_format)})', fontsize="18")
plt.yticks(np.arange(round(df_wg_usd_1year["usd"].min(), 1), round(df_wg_usd_1year["usd"].max() + 0.1, 1), 0.1), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b, %Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=29))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Câmbio USD ↔ BRL", fontsize="18")
plt.legend(fontsize=18, loc="upper left", bbox_to_anchor=(0.02, 0, 0, 0.93))
plt.show()

# Boxplot for 5 years:

plt.figure(figsize=(15,10), dpi=300)
sns.boxplot(x=df_wg_usd_5year["usd"], palette="viridis")
plt.title("Boxplot do Dataset - 5 anos (1826 dias)", fontsize="18")
plt.show()

# Boxplot for 1 year:

sns.set(palette="viridis")
sns.set_style("whitegrid",  {"grid.linestyle": ":"})
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15,12), dpi=600)
ax1.set_title("Boxplot - Dados originais", fontsize="22")
sns.boxplot(data=df_wg_usd_1year["usd"], ax=ax1, orient="v")
sns.stripplot(data=df_wg_usd_1year["usd"], ax=ax1, jitter=0.1, size=12, alpha=0.5)
ax2.set_title("Boxplot - Dados diferenciados - 1 Ordem", fontsize="22")
sns.boxplot(data=df_wg_usd_1year["diff"].dropna(), ax=ax2, orient="v")
sns.stripplot(data=df_wg_usd_1year["diff"].dropna(), ax=ax2, jitter=0.1, size=12, alpha=0.5)
plt.show()

# In[0.4]: Calculate Statistics for datasets

var_usd_1year = np.var(df_wg_usd_1year['usd'])
var_usd_1year

# 0.02150862174375002

varlog_usd_1year = np.var(df_wg_usd_1year['log'])
varlog_usd_1year

# 0.0008225663376648036

optimal_lags_usd_1year = 12*(len(df_wg_usd_1year['usd'])/100)**(1/4)
optimal_lags_usd_1year

# 15.961265820308224

# So optimal lags for this dataset = 14 or 15

# In[0.6]: Plot the original dataset for visual assessment

plt.figure(figsize=(15, 10), dpi=300)
sns.scatterplot(x=usd_train["dateTime"], y=usd_train["usd"], color="limegreen", label="Câmbio USD")
plt.axhline(y=np.mean(usd_train["usd"]), color="black", linestyle="--", label="Média", linewidth=2) # mean for usd
plt.axhline(y=np.max(usd_train["usd"]), color="magenta", label="Máxima", linewidth=2) # max for usd
plt.axhline(y=np.min(usd_train["usd"]), color="magenta", linestyle="--", label="Mínima", linewidth=2) # min for usd
plt.title(f'Cotação do Dólar - Série histórica ({usd_train["dateTime"][0].strftime(dt_format)} - {usd_train["dateTime"][239].strftime(dt_format)})', fontsize="18")
plt.yticks(np.arange(round(usd_train["usd"].min(), 1), round(usd_train["usd"].max() + 0.1, 1), 0.1), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b, %Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=29))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Câmbio USD ↔ BRL", fontsize="18")
plt.legend(fontsize=18, loc="upper left", bbox_to_anchor=(0.02, 0, 0, 0.93))
plt.show()

# In[0.7]: Plot the log dataset for visual assessment

plt.figure(figsize=(15, 10), dpi=300)
sns.lineplot(x=usd_train["dateTime"], y=df_wg_usd_1year["log"], color="limegreen", label="log(Câmbio USD)")
plt.axhline(y=np.mean(usd_train["log"]), color="black", linestyle="--", label="Média", linewidth=2) # mean for usd
plt.axhline(y=np.max(usd_train["log"]), color="magenta", label="Máxima", linewidth=2) # max for usd
plt.axhline(y=np.min(usd_train["log"]), color="magenta", linestyle="--", label="Mínima", linewidth=2) # min for usd
plt.title(f'Cotação do Dólar - Série histórica ({usd_train["dateTime"][0].strftime(dt_format)} - {usd_train["dateTime"][239].strftime(dt_format)})', fontsize="18")
plt.yticks(np.arange(round(usd_train["log"].min(), 2), round(usd_train["log"].max() + 0.01, 2), 0.01), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b, %Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=29))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Câmbio log(USD ↔ BRL)", fontsize="18")
plt.legend(fontsize=18, loc="upper left", bbox_to_anchor=(0.02, 0, 0, 0.93))
plt.show()


# In[1.0]: Defining Stationarity

# In[1.1]: Augmented Dickey-Fuller Test with usd and log

usd_1year_adf_ols = adfuller(usd_train['usd'], maxlag=None, autolag="AIC", store=True, regresults=True)[-1].resols.summary()
usd_1year_adf_ols

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.001
Model:                            OLS   Adj. R-squared:                 -0.003
Method:                 Least Squares   F-statistic:                    0.2327
Date:                Sun, 29 Sep 2024   Prob (F-statistic):              0.630
Time:                        13:35:40   Log-Likelihood:                 587.49
No. Observations:                 297   AIC:                            -1171.
Df Residuals:                     295   BIC:                            -1164.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1            -0.0038      0.008     -0.482      0.630      -0.019       0.012
const          0.0219      0.041      0.538      0.591      -0.058       0.102
==============================================================================
Omnibus:                       13.419   Durbin-Watson:                   2.045
Prob(Omnibus):                  0.001   Jarque-Bera (JB):               27.783
Skew:                           0.166   Prob(JB):                     9.27e-07
Kurtosis:                       4.461   Cond. No.                         111.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

# Doing the same for the log database

log_1year_adf_ols = adfuller(usd_train['log'], maxlag=None, autolag="AIC", store=True, regresults=True)[-1].resols.summary()
log_1year_adf_ols

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.001
Model:                            OLS   Adj. R-squared:                 -0.003
Method:                 Least Squares   F-statistic:                    0.2527
Date:                Sun, 29 Sep 2024   Prob (F-statistic):              0.616
Time:                        13:36:01   Log-Likelihood:                 1079.7
No. Observations:                 297   AIC:                            -2155.
Df Residuals:                     295   BIC:                            -2148.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1            -0.0040      0.008     -0.503      0.616      -0.020       0.012
const          0.0069      0.013      0.536      0.592      -0.018       0.032
==============================================================================
Omnibus:                       10.349   Durbin-Watson:                   2.049
Prob(Omnibus):                  0.006   Jarque-Bera (JB):               17.910
Skew:                           0.162   Prob(JB):                     0.000129
Kurtosis:                       4.159   Cond. No.                         78.1
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

# We can notice that in the second model, both AIC and BIC are lower than the first one
# and also that the loglike is higher on the second model than the first one, proving
# that a logarthmic version of the data is a better fit for the model so far.

log_1year_adf = adfuller(usd_train["log"], maxlag=None, autolag="AIC")
log_1year_adf

# p-value is > 0.05 (0.891), DO NOT REJECT the null hypothesis and suggests the presence of a unit root (NON-STATIONARY)
# adf stats > 10% (-0.5), DO NOT REJECT the null hypothesis and suggests the presence of a unit root (NON-STATIONARY)
# lags = 0 good amount of lags

"""
(-0.5026776161124509,
 0.8914929176801636,
 0,
 297,
 {'1%': -3.4525611751768914,
  '5%': -2.87132117782556,
  '10%': -2.5719816428028888},
 -2046.0993933361387)
"""

# In[1.2]: Running KPSS test to determine stationarity for usd

log_1year_kpss = kpss(usd_train['log'], regression="c", nlags="auto")
log_1year_kpss

# p-value < 0.05 (0.01) REJECTS the null hypothesis, suggesting data is NON-STATIONARY
# kpss stat of 2.19 > 1% (0.739), REJECTS the null hypothesis, suggesting data is NON-STATIONARY
# lags = 10 good amount of lags

"""
(2.194652093125916,
 0.01,
 10,
 {'10%': 0.347, '5%': 0.463, '2.5%': 0.574, '1%': 0.739})
"""

# In[1.3]: Plotting ACF and PACF to examine autocorrelations in data:

fig, (ax1, ax2) = plt.subplots(2, figsize=(11,7), dpi=300)
plot_acf(usd_train["log"], ax=ax1)
plot_pacf(usd_train["log"], ax=ax2)
plt.show()

# PACF plot shows an AR(2) order for the dataset showing a high statistical significance spike
# at the first lag in PACF. ACF shows slow decay towards 0 -> exponentially decaying or sinusoidal

# In[1.4]: Defining the order of differencing we need:

# Original Series
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(9,7), dpi=300)
ax1.plot(usd_train["log"]);
ax1.set_title('Série Original log(USD)');
ax1.axes.xaxis.set_visible(False)

# 1st Differencing
ax2.plot(usd_train["diff"].dropna());
ax2.set_title('1ª Ordem de Diferenciação');
ax2.axes.xaxis.set_visible(False)

# 2nd Differencing
ax3.plot(usd_train["diff"].diff().dropna());
ax3.set_title('2ª Ordem de Diferenciação') 
plt.show()

# Plotting the ACF for each order

fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(9,7), dpi=300)
plot_acf(usd_train["log"], ax=ax1)
plot_acf(usd_train["diff"].dropna(), ax=ax2)
plot_acf(usd_train["diff"].diff().dropna(), ax=ax3)
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

# Plotting ACF and PACF together for the first order diff:

fig, (ax1, ax2) = plt.subplots(2, figsize=(11,8), dpi=300)
plot_acf(usd_train["diff"].dropna(), ax=ax1)
plot_pacf(usd_train["diff"].dropna(), ax=ax2)
plt.suptitle("ACF e PACF - USD - 1ª Ordem de Diferenciação", fontsize="18")
plt.show()

# These plots show a sharp cut off at ACF lag 2, which indicates sligh overdifferencing
# and also an MA parameter of 2. The stationarized series display an "MA signature"
# Meaning we can explain the autocorrelation pattern my adding MA terms rather than
# AR terms. PACF is related to AR (orders while ACF is related to MA (lags of the forecast
# errors). We can see from the PACF plot that after the first lag we don't have
# any other positive lags outside of the significance limit, so we 

# We can see a pretty good visually STATIONARY plot on the first differenciation,
# so we'll be going on with a single differenciation and we'll check how that looks
# with ACF and PACF

# In[1.5]: Plotting the differenced data for visual assessment:

plt.figure(figsize=(15, 10))
sns.lineplot(x=usd_train["dateTime"], y=usd_train["diff"], color="limegreen", label="Log(USD) - Diff")
plt.axhline(y=np.mean(usd_train["diff"]), color="black", linestyle="--", label="Média") # mean for usd
plt.title(f'Log do Dólar Diferenciado - Série histórica ({usd_train["dateTime"][0].strftime(dt_format)} - {usd_train["dateTime"][279].strftime(dt_format)})', fontsize="18")
plt.yticks(np.arange(round(usd_train["diff"].min(), 2), round(usd_train["diff"].max() + 0.01, 2), 0.02), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b, %Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=29))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Log(Câmbio USD ↔ BRL) Diferenciado ", fontsize="18")
plt.legend(fontsize=18, loc="upper center")
plt.show()

# In[2.0]: Defining Stationarity again

optimal_lags_usd_train = 12*(len(usd_train['diff'].dropna())/100)**(1/4)
optimal_lags_usd_train

# 15.753257007058663

# So optimal lags for this dataset = 15 or 16


# In[2.1]: Augmented Dickey-Fuller Test with diff

log_1year_diff_adf = adfuller(usd_train["diff"].dropna(), maxlag=None, autolag="AIC")
log_1year_diff_adf

# p-value is << 0.05 (1.668561287686211e-10), REJECTS the null hypothesis and suggests the absence of a unit root (STATIONARY)
# adf stats << 1% (-7), REJECTS the null hypothesis and suggests the absence of a unit root (STATIONARY)
# lags = 0 good amount of lags

"""
(-17.589441180972766,
 3.970709144117753e-30,
 0,
 296,
 {'1%': -3.452636878592149,
  '5%': -2.8713543954331433,
  '10%': -2.5719993576515705},
 -1108.6017967128407)
"""

log_1year_diff_adf_ols = adfuller(usd_train['diff'].dropna(), maxlag=None, autolag="AIC", store=True, regresults=True)[-1].resols.summary()
log_1year_diff_adf_ols

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.513
Model:                            OLS   Adj. R-squared:                  0.511
Method:                 Least Squares   F-statistic:                     309.4
Date:                Sun, 29 Sep 2024   Prob (F-statistic):           8.14e-48
Time:                        13:51:22   Log-Likelihood:                 584.99
No. Observations:                 296   AIC:                            -1166.
Df Residuals:                     294   BIC:                            -1159.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1            -1.0255      0.058    -17.589      0.000      -1.140      -0.911
const          0.0024      0.002      1.201      0.231      -0.002       0.006
==============================================================================
Omnibus:                       12.880   Durbin-Watson:                   2.001
Prob(Omnibus):                  0.002   Jarque-Bera (JB):               27.330
Skew:                           0.129   Prob(JB):                     1.16e-06
Kurtosis:                       4.466   Cond. No.                         29.8
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

# The note[2] in the OLS Regression is worrisome, but we can mitigate that by running
# these tests again but setting a finite amount lags. This might increase the test
# stat and decrease the value of the p-value, making it seem more over-fitted, but
# this will leave it like the eur dataset basically. Let's hope the kpss also shows
# a "normal" result.
# This run also shows no statistical improvement  in AIC and BIC from the last OLS
# Regression, and it actually shows the Loglike is slighly lower in this one.

# This time it gave us a better result for the OLS, it also gave better results for the
# AIC, BIC and Loglike stats, making this a better model than before.

# In[2.3]: Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test post differencing

log_1year_diff_kpss = kpss(usd_train['diff'].dropna(), regression="c", nlags="auto")
log_1year_diff_kpss

# p-value > 0.05 (0.1) DO NOT REJECT the null hypothesis, suggesting data is STATIONARY
# kpss stat of 0.13 is < 10% (0.347), DO NOT REJECT the null hypothesis, suggesting data is STATIONARY
# lags = 6 ok amount of lags for this point in time

"""
(0.1381961585361327,
 0.1,
 6,
 {'10%': 0.347, '5%': 0.463, '2.5%': 0.574, '1%': 0.739})
"""

# The adf test shows similar results to the ones for EURO, which could mean a bad time.
# KPSS though shows almost better (?) results than the EURO one, but still validates stationarity
# although it's veeery close to not being stationary as both the p-value and kpss stats are
# very close to the limit. This also has a high number of lags, but it's ok.

# In[2.4]: Plotting ACF and PACF to determine correct number of lags for usd

fig, (ax1, ax2) = plt.subplots(2, figsize=(11, 8), dpi=300)
plot_acf(df_wg_usd_1year['diff'].dropna(), ax=ax1, lags=13)
plot_pacf(df_wg_usd_1year['diff'].dropna(), ax=ax2, lags=13)
plt.show()

# Both tests seem to show normal stationary plots with no significant lags after zero.
# Also, no significant negative first lag;

# In[2.5]: Testing the lagged data for multicolinearity:

# Creating a new DataFrame with the lagged values:

usd_1year_lagged = pd.DataFrame({"log": usd_train["log"]})

usd_1year_lagged['lag 1'] = usd_1year_lagged["log"].shift(1)
usd_1year_lagged['lag 2'] = usd_1year_lagged["log"].shift(2)
usd_1year_lagged['lag 3'] = usd_1year_lagged["log"].shift(3)
usd_1year_lagged['lag 4'] = usd_1year_lagged["log"].shift(4)
usd_1year_lagged['lag 5'] = usd_1year_lagged["log"].shift(5)
usd_1year_lagged['lag 6'] = usd_1year_lagged["log"].shift(6)
usd_1year_lagged['lag 7'] = usd_1year_lagged["log"].shift(7)

usd_1year_lagged

usd_constants = add_constant(usd_1year_lagged.dropna())

usd_vif = pd.DataFrame()

usd_vif['vif'] = [variance_inflation_factor(usd_constants.values, i) for i in range(usd_constants.shape[1])]

usd_vif['variable'] = usd_constants.columns

usd_vif

"""
           vif variable
0  1271.152045    const
1    56.966101      log
2   108.575289    lag 1
3   105.158174    lag 2
4   103.898957    lag 3
5   103.611491    lag 4
6   102.401394    lag 5
7   103.306870    lag 6
8    53.832879    lag 7
"""

# We can see here a lot of high VIFs, but this is expected, since these values
# come from just lagged datasets they are bound to have multicolinearity, which
# is not necessarily a problem, we just need to be cautions to the amount of
# lagged exogs we add at the end and check AIC and BIC

# In[3.0]: Running Granger Causality tests to analyse the lags

# In[3.0]: Runnning the actual Causality test on the lagged data

usd_granger_1 = grangercausalitytests(usd_1year_lagged[["log", "lag 1"]].dropna(), maxlag=7)

usd_granger_1

# No relevant lags for usd

# In[3.2]: Testing Granger Causality with "diff" and "eur"

eur_1year_granger = pd.read_csv("./datasets/wrangled/df_eur_1year.csv", float_precision="high", parse_dates=([0]))

eur_1year_granger = eur_1year_granger.drop(eur_1year_granger[eur_1year_granger["weekday"] == "Sunday"].index)[:-15]

df_usd_eur = pd.DataFrame({"usd": usd_train['usd'], "eur": eur_1year_granger["eur"]})

usd_eur_granger = grangercausalitytests(df_usd_eur[['usd', 'eur']].dropna(), maxlag=40)

"""
Granger Causality
number of lags (no zero) 6
ssr based F test:         F=2.9961  , p=0.0075  , df_denom=279, df_num=6
ssr based chi2 test:   chi2=18.8140 , p=0.0045  , df=6
likelihood ratio test: chi2=18.2327 , p=0.0057  , df=6
parameter F test:         F=2.9961  , p=0.0075  , df_denom=279, df_num=6
"""


"""
Granger Causality
number of lags (no zero) 11
ssr based F test:         F=2.3593  , p=0.0086  , df_denom=264, df_num=11
ssr based chi2 test:   chi2=28.2128 , p=0.0030  , df=11
likelihood ratio test: chi2=26.9108 , p=0.0047  , df=11
parameter F test:         F=2.3593  , p=0.0086  , df_denom=264, df_num=11
"""

# Other relevant lags: 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 22, 23, 24
# 25, 26. 40

# It looks like Euro could be a great predictor for USD. There are significant
# predictor values from lag 4 to lag 26. This sounds insane. I'll check these
# in the ARIMA models and get the best by using AIC and BIC.

# In[2.5]: Testing the eur and usd data for multicolinearity:

usd_eur_vif_test = pd.DataFrame({"usd": usd_train["log"], "eur": eur_1year_granger["log"]})

usd_eur_vif_test

usd_constants = add_constant(usd_eur_vif_test.dropna())

usd_eur_vif = pd.DataFrame()

usd_eur_vif['vif'] = [variance_inflation_factor(usd_constants.values, i) for i in range(usd_constants.shape[1])]

usd_eur_vif['variable'] = usd_constants.columns

usd_eur_vif

"""
           vif variable
0  1201.455879    const
1    15.258933      log
2    15.258933   logEUR
"""

# There's a high VIF between these two datasets, but since these are close to VIF 10
# I'll still try to fit them in the model and check the statistics

# In[3.1]: Cross-testing "usd" and "eur"
# to make sure they have causality between them

tscv = TimeSeriesSplit(n_splits=5)

ct_usd_eur = usd_eur_vif_test.dropna()

for train_index, test_index in tscv.split(ct_usd_eur):
    train, test = ct_usd_eur.iloc[train_index], ct_usd_eur.iloc[test_index]

    X_train, y_train = train['eur'], train['usd']
    X_test, y_test = test['eur'], test['usd']

    granger_result = grangercausalitytests(train[['usd', 'eur']], maxlag=10, verbose=False)

    for lag, result in granger_result.items():
        f_test_pvalue = result[0]['ssr_ftest'][1]  # p-value from F-test
        print(f"Lag: {lag}, P-value: {f_test_pvalue}")

    print(f"TRAIN indices: {train_index}")
    print(f"TEST indices: {test_index}")

"""
Lag: 1, P-value: 0.07894970834040042
Lag: 2, P-value: 0.17214629358804942
Lag: 3, P-value: 0.1889652593904693
Lag: 4, P-value: 0.2859395374996287

Lag: 1, P-value: 0.03757678093567854
Lag: 2, P-value: 0.10038461796563021
Lag: 3, P-value: 0.1927356017858318
Lag: 4, P-value: 0.36376954865999195

Lag: 1, P-value: 0.04775370943247826
Lag: 2, P-value: 0.1373544140546217
Lag: 3, P-value: 0.25152967098272955
Lag: 4, P-value: 0.4651441112842434

Lag: 1, P-value: 0.39859615139198734
Lag: 2, P-value: 0.6263421890920511
Lag: 3, P-value: 0.4330740050990468
Lag: 4, P-value: 0.21595677666048627

Lag: 1, P-value: 0.6874837899885218
Lag: 2, P-value: 0.8059615108205229
Lag: 3, P-value: 0.6273303752069179
Lag: 4, P-value: 0.09337104957105065
"""

# The test shows possible prediction power addition from the eur dataset in the first lag

# In[4.0]: Save final dataset for testing ARIMA

usd_train_1year_365 = pd.DataFrame({
    "date": usd_train["dateTime"],
    "usd": usd_train["usd"],
    "log": usd_train["log"],
    "diff": usd_train["diff"],
    "eur": eur_1year_granger["log"],
    })

usd_test_1year_365 = usd_test

# save to csv
usd_train_1year_365.to_csv("./datasets/arima_ready/usd_train_1year_365.csv", index=False)
usd_test_1year_365.to_csv("./datasets/arima_ready/usd_test_1year_365.csv", index=False)
