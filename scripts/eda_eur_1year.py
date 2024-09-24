# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # #
# Exploratory Data Analysis = EURO - 1 year #
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

df_wg_eur_1year = pd.read_csv("./datasets/wrangled/df_eur_1year.csv", float_precision="high", parse_dates=([0]))

df_wg_eur_1year.info()

df_wg_eur_5year = pd.read_csv("./datasets/wrangled/df_eur_5year.csv", float_precision="high", parse_dates=([0]))

df_wg_eur_5year

dt_format = "%d/%m/%Y"

# In[0.3]: Plot the 5 year dataset

plt.figure(figsize=(15, 10))
sns.lineplot(x=df_wg_eur_5year["dateTime"], y=df_wg_eur_5year["eur"], color="limegreen", label="Câmbio EUR")
plt.axhline(y=np.mean(df_wg_eur_5year["eur"]), color="black", linestyle="--", label="Média") # mean for euro
plt.axhline(y=np.max(df_wg_eur_5year["eur"]), color="magenta", label="Máxima") # máxima for euro
plt.axhline(y=np.min(df_wg_eur_5year["eur"]), color="magenta", linestyle="--", label="Mínima") # máxima for euro
plt.title(f'Cotação do Euro - Série histórica ({df_wg_eur_5year["dateTime"][0].strftime(dt_format)} - {df_wg_eur_5year["dateTime"][1740].strftime(dt_format)})', fontsize="18")
plt.yticks(np.arange(round(df_wg_eur_5year["eur"].min(), 1), round(df_wg_eur_5year["eur"].max() + 0.1, 1), 0.1), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b, %Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=90))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Câmbio EUR ↔ BRL", fontsize="18")
plt.legend(fontsize=18, loc="upper right", bbox_to_anchor=(0.98, 0, 0, 0.93))
plt.show()

# In[0.4]: Calculate Statistics for datasets

var_eur_1year = np.var(df_wg_eur_1year['eur'])
var_eur_1year

#0.020617129582265307

varlog_eur_1year = np.var(df_wg_eur_1year['logEUR'])
varlog_eur_1year

#0.0006763790454904603

optimal_lags_eur_1year = 12*(len(df_wg_eur_1year['eur'])/100)**(1/4)
optimal_lags_eur_1year

# 15.522824731401618

# So optimal lags for this dataset = 15 or 16

# In[0.5]: Plot the original dataset for visual assessment

plt.figure(figsize=(15, 10))
sns.lineplot(x=df_wg_eur_1year["dateTime"], y=df_wg_eur_1year["eur"], color="limegreen", label="Câmbio EUR")
plt.axhline(y=np.mean(df_wg_eur_1year["eur"]), color="black", linestyle="--", label="Média") # mean for euro
plt.axhline(y=np.max(df_wg_eur_1year["eur"]), color="magenta", label="Máxima") # máxima for euro
plt.axhline(y=np.min(df_wg_eur_1year["eur"]), color="magenta", linestyle="--", label="Mínima") # máxima for euro
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

# In[0.6]: Plot the log dataset for visual assessment

plt.figure(figsize=(15, 10))
sns.scatterplot(x=df_wg_eur_1year["dateTime"], y=df_wg_eur_1year["eur"], color="limegreen", label="Câmbio EUR")
plt.axhline(y=np.mean(df_wg_eur_1year["eur"]), color="black", linestyle="--", label="Média") # mean for euro
plt.axhline(y=np.max(df_wg_eur_1year["eur"]), color="magenta", label="Máxima") # máxima for euro
plt.axhline(y=np.min(df_wg_eur_1year["eur"]), color="magenta", linestyle="--", label="Mínima") # máxima for euro
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

# In[0.7]: Plot the log dataset for visual assessment

plt.figure(figsize=(15, 10))
sns.lineplot(x=df_wg_eur_1year["dateTime"], y=df_wg_eur_1year["logEUR"], color="limegreen", label="log(Câmbio EUR)")
plt.axhline(y=np.mean(df_wg_eur_1year["logEUR"]), color="black", linestyle="--", label="Média") # mean for euro
plt.axhline(y=np.max(df_wg_eur_1year["logEUR"]), color="magenta", label="Máxima") # máxima for euro
plt.axhline(y=np.min(df_wg_eur_1year["logEUR"]), color="magenta", linestyle="--", label="Mínima") # máxima for euro
plt.title(f'Cotação do Euro - Série histórica ({df_wg_eur_1year["dateTime"][0].strftime(dt_format)} - {df_wg_eur_1year["dateTime"][279].strftime(dt_format)})', fontsize="18")
plt.yticks(np.arange(round(df_wg_eur_1year["logEUR"].min(), 2), round(df_wg_eur_1year["logEUR"].max() + 0.01, 2), 0.01), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b, %Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=29))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Câmbio log(EUR ↔ BRL)", fontsize="18")
plt.legend(fontsize=18, loc="upper left", bbox_to_anchor=(0.02, 0, 0, 0.93))
plt.show()


# In[1.0]: Defining Stationarity

# In[1.1]: Augmented Dickey-Fuller Test with eur and logEUR

eur_1year_adf_ols = adfuller(df_wg_eur_1year['eur'], maxlag=None, autolag="AIC", store=True, regresults=True)[-1].resols.summary()
eur_1year_adf_ols

# Normal values results:
"""
                            OLS Regression Results
==============================================================================
Dep. Variable:                      y   R-squared:                       0.050
Model:                            OLS   Adj. R-squared:                  0.021
Method:                 Least Squares   F-statistic:                     1.716
Date:                Sat, 21 Sep 2024   Prob (F-statistic):             0.0949
Time:                        12:43:23   Log-Likelihood:                 600.30
No. Observations:                 272   AIC:                            -1183.
Df Residuals:                     263   BIC:                            -1150.
Df Model:                           8
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1             0.0174      0.013      1.356      0.176      -0.008       0.043
x2            -0.1177      0.063     -1.859      0.064      -0.242       0.007
x3            -0.1457      0.063     -2.298      0.022      -0.271      -0.021
x4            -0.1075      0.064     -1.674      0.095      -0.234       0.019
x5            -0.1057      0.064     -1.649      0.100      -0.232       0.020
x6            -0.0173      0.064     -0.271      0.787      -0.143       0.109
x7            -0.0698      0.064     -1.099      0.273      -0.195       0.055
x8            -0.1323      0.063     -2.096      0.037      -0.257      -0.008
const         -0.0911      0.070     -1.308      0.192      -0.228       0.046
==============================================================================
Omnibus:                       24.943   Durbin-Watson:                   1.988
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               65.716
Skew:                           0.378   Prob(JB):                     5.37e-15
Kurtosis:                       5.286   Cond. No.                         314.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

# Doing the same for the log database

logeur_1year_adf_ols = adfuller(df_wg_eur_1year['logEUR'], maxlag=None, autolag="AIC", store=True, regresults=True)[-1].resols.summary()
logeur_1year_adf_ols

# Results for log values:
"""
                            OLS Regression Results
==============================================================================
Dep. Variable:                      y   R-squared:                       0.048
Model:                            OLS   Adj. R-squared:                  0.019
Method:                 Least Squares   F-statistic:                     1.658
Date:                Sat, 21 Sep 2024   Prob (F-statistic):              0.109
Time:                        17:40:55   Log-Likelihood:                 1062.5
No. Observations:                 272   AIC:                            -2107.
Df Residuals:                     263   BIC:                            -2075.
Df Model:                           8
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1             0.0156      0.013      1.207      0.228      -0.010       0.041
x2            -0.1134      0.063     -1.795      0.074      -0.238       0.011
x3            -0.1443      0.063     -2.282      0.023      -0.269      -0.020
x4            -0.1047      0.064     -1.637      0.103      -0.231       0.021
x5            -0.0963      0.064     -1.508      0.133      -0.222       0.029
x6            -0.0198      0.064     -0.311      0.756      -0.145       0.106
x7            -0.0626      0.063     -0.990      0.323      -0.187       0.062
x8            -0.1340      0.063     -2.131      0.034      -0.258      -0.010
const         -0.0258      0.022     -1.180      0.239      -0.069       0.017
==============================================================================
Omnibus:                       24.280   Durbin-Watson:                   1.990
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               62.554
Skew:                           0.373   Prob(JB):                     2.61e-14
Kurtosis:                       5.228   Cond. No.                         553.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

# We can notice that in the second model, both AIC and BIC are lower than the first one
# and also that the loglike is higher on the second model than the first one, proving
# that a logarthmic version of the data is a better fit for the model so far.

# Getting values from the second model:

logeur_1year_adf = adfuller(df_wg_eur_1year["logEUR"], maxlag=None, autolag="AIC")
logeur_1year_adf

# p-value is > 0.05 (0.99), DO NOT REJECT the null hypothesis and suggests the presence of a unit root (NON-STATIONARY)
# adf stats > 10% (1.20), DO NOT REJECT the null hypothesis and suggests the presence of a unit root (NON-STATIONARY)
# lags = 7 good amount of lags

"""
(1.207357376225017,
 0.9960363235388519,
 7,
 272,
 {'1%': -3.4546223782586534,
  '5%': -2.8722253212300277,
  '10%': -2.5724638500216264},
 -2048.618823297598)
"""

# In[1.2]: Running KPSS test to determine stationarity for eur

logeur_1year_kpss = kpss(df_wg_eur_1year['logEUR'], regression="c", nlags="auto")
logeur_1year_kpss

# p-value < 0.05 (0.01) REJECTS the null hypothesis, suggesting data is NON-STATIONARY
# kpss stat of 2.00 is > than 1% (0.739), REJECTS the null hypothesis, suggesting data is NON-STATIONARY
# lags = 10 good amount of lags

"""
(2.004541680328264,
 0.01,
 10,
 {'10%': 0.347, '5%': 0.463, '2.5%': 0.574, '1%': 0.739})
"""

# In[1.3]: Plotting ACF and PACF to examine autocorrelations in data:

plt.figure(figsize=(12,6))
plot_acf(df_wg_eur_1year['logEUR'].dropna(), lags=13)
plt.show()

plt.figure(figsize=(12,6))
plot_pacf(df_wg_eur_1year['logEUR'].dropna(), lags=13)
plt.show()

# PACF plot shows an AR(2) order for the dataset showing a high statistical significance spike 
# at the first lag in PACF. ACF shows slow decay towards 0 -> exponentially decaying or sinusoidal

# In[1.4]: Defining the order of differencing we need:

plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120}) 

# Original Series
fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot(df_wg_eur_1year["logEUR"]);
ax1.set_title('Série Original log(EUR)');
ax1.axes.xaxis.set_visible(False)

# 1st Differencing
ax2.plot(df_wg_eur_1year["logEUR"].diff());
ax2.set_title('1ª Ordem de Diferenciação');
ax2.axes.xaxis.set_visible(False)

# 2nd Differencing
ax3.plot(df_wg_eur_1year["logEUR"].diff().diff());
ax3.set_title('2ª Ordem de Diferenciação') 
plt.show()


# Plotting the ACF for each order

plt.rcParams.update({'figure.figsize':(11,10), 'figure.dpi':120})

fig, (ax1, ax2, ax3) = plt.subplots(3)
plot_acf(df_wg_eur_1year["logEUR"], ax=ax1)
plot_acf(df_wg_eur_1year["logEUR"].diff().dropna(), ax=ax2)
plot_acf(df_wg_eur_1year["logEUR"].diff().diff().dropna(), ax=ax3)

# We can see a pretty good visually STATIONARY plot on the first differenciation,
# We can also see, after the first lag in the first diff ACF we have a pretty good
# white noise plot, and although we can see that the second lag goes into negative,
# meaning we could have a little overdifferentiation, we can deal with that in the
# final model by adding a number of MA terms at the final ARIMA. while in the
# second differentiation (second ACF) plot we can see that the second lag it goes
# straight and significantly into negative values, indicating a lot of over
# differentiation. I'll be going on with a single differenciation and I'll be
# looking for a number of MAs in the PACF plots next

# Plotting the ACf for the first order diff:
plt.rcParams.update({'figure.figsize':(9,5), 'figure.dpi':120})
    
plot_pacf(df_wg_eur_1year['logEUR'].diff().dropna())
plt.show()

# Plotting ACF and PACF together:

plt.rcParams.update({'figure.figsize':(11,8), 'figure.dpi':120})

fig, (ax1, ax2) = plt.subplots(2)
plot_acf(df_wg_eur_1year["logEUR"].diff().dropna(), ax=ax1)
plot_pacf(df_wg_eur_1year["logEUR"].diff().dropna(), ax=ax2)

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
 
# In[1.5]: Differencing the data to achieve stationarity

df_wg_eur_1year['diffLogEUR'] = df_wg_eur_1year['logEUR'].diff()

df_wg_eur_1year

# In[1.6]: Plotting the differenced data for visual assessment:

plt.figure(figsize=(15, 10))
sns.lineplot(x=df_wg_eur_1year["dateTime"], y=df_wg_eur_1year["diffLogEUR"], color="limegreen", label="Log(EUR) - Diff")
plt.axhline(y=np.mean(df_wg_eur_1year["diffLogEUR"]), color="black", linestyle="--", label="Média") # mean for eur
plt.title(f'Log do Euro Diferenciado - Série histórica ({df_wg_eur_1year["dateTime"][0].strftime(dt_format)} - {df_wg_eur_1year["dateTime"][279].strftime(dt_format)})', fontsize="18")
plt.yticks(np.arange(round(df_wg_eur_1year["diffLogEUR"].min(), 3), round(df_wg_eur_1year["diffLogEUR"].max() + 0.002, 3), 0.002), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b, %Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=29))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Log(Câmbio EUR ↔ BRL) Diferenciado ", fontsize="18")
plt.legend(fontsize=18, loc="upper center")
plt.show()

# In[2.0]: Defining Stationarity again

optimal_lags_eur_1year_diff = 12*(len(df_wg_eur_1year['diffLogEUR'].dropna())/100)**(1/4)
optimal_lags_eur_1year_diff

# 15.508946465645465

# So optimal lags for this dataset = 15 or 16

# In[2.1]: Augmented Dickey-Fuller Test with diffLogEUR

logeur_1year_diff_adf = adfuller(df_wg_eur_1year["diffLogEUR"].dropna(), maxlag=None, autolag="AIC")
logeur_1year_diff_adf

# p-value is <<<< 0.05 (5.94330005090518e-25), REJECTS the null hypothesis and suggests the absence of a unit root (STATIONARY)
# adf stats <<<< 1% (-13), REJECTS the null hypothesis and suggests the absence of a unit root (STATIONARY)
# lags = 1 good amount of lags

"""
(-13.340116766071626,
 5.94330005090518e-25,
 1,
 277,
 {'1%': -3.4541800885158525,
  '5%': -2.872031361137725,
  '10%': -2.5723603999791473},
 -2040.329889589716)
"""

logeur_1year_diff_adf_ols = adfuller(df_wg_eur_1year['diffLogEUR'].dropna(), maxlag=None, autolag="AIC", store=True, regresults=True)[-1].resols.summary()
logeur_1year_diff_adf_ols

"""
                            OLS Regression Results
==============================================================================
Dep. Variable:                      y   R-squared:                       0.536
Model:                            OLS   Adj. R-squared:                  0.533
Method:                 Least Squares   F-statistic:                     158.3
Date:                Sun, 22 Sep 2024   Prob (F-statistic):           2.04e-46
Time:                        12:15:14   Log-Likelihood:                 1079.4
No. Observations:                 277   AIC:                            -2153.
Df Residuals:                     274   BIC:                            -2142.
Df Model:                           2
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1            -1.1908      0.089    -13.340      0.000      -1.367      -1.015
x2             0.1141      0.061      1.874      0.062      -0.006       0.234
const          0.0005      0.000      1.623      0.106      -0.000       0.001
==============================================================================
Omnibus:                       25.480   Durbin-Watson:                   2.001
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               64.763
Skew:                           0.397   Prob(JB):                     8.65e-15
Kurtosis:                       5.232   Cond. No.                         343.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

# AIC, BIC and Loglike prove this model is even better than the second one, so
# I SHOULD USE D = 1 IN THE ARIMA MODEL

# In[2.2]: Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test post differencing

logeur_1year_diff_kpss = kpss(df_wg_eur_1year['diffLogEUR'].dropna(), regression="c", nlags="auto")
logeur_1year_diff_kpss

# p-value > 0.05 (0.1) DO NOT REJECT the null hypothesis, suggesting data is STATIONARY
# kpss stat of 0.28 is < 10% (0.347), DO NOT REJECT the null hypothesis, suggesting data is STATIONARY
# lags = 10 ok amount of lags for this point in time

"""
(0.2825087625709679,
 0.1,
 10,
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

plt.figure(figsize=(12,6))
plot_acf(df_wg_eur_1year['diffLogEUR'].dropna(), lags=13)
plt.show()

plt.figure(figsize=(12,6))
plot_pacf(df_wg_eur_1year['diffLogEUR'].dropna(), lags=13)
plt.show()

# Both tests seem to show normal stationary plots with no significant lags after zero.
# Also, no significant negative first lag;

# In[3.0]: Running Granger Causality tests

# In[3.1]: Create the the lagged dataframe:

eur_1year_lagged = pd.DataFrame({"logEUR": df_wg_eur_1year["logEUR"]})

eur_1year_lagged['lag 1'] = eur_1year_lagged["logEUR"].shift(1)
eur_1year_lagged['lag 2'] = eur_1year_lagged["logEUR"].shift(2)
eur_1year_lagged['lag 3'] = eur_1year_lagged["logEUR"].shift(3)
eur_1year_lagged['lag 4'] = eur_1year_lagged["logEUR"].shift(4)
eur_1year_lagged['lag 5'] = eur_1year_lagged["logEUR"].shift(5)
eur_1year_lagged['lag 6'] = eur_1year_lagged["logEUR"].shift(6)

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
0  4966.815867    const
1    27.743165   logEUR
2    49.482221    lag 1
3    48.138446    lag 2
4    47.919188    lag 3
5    46.892520    lag 4
6    45.948085    lag 5
7    25.236549    lag 6
"""

# We can see very high VIFs between the lags; Which means there's multicolinearity

# In[3.3]: Running the actual Causality test on the lagged data

eur_granger_lag2 = grangercausalitytests(eur_1year_lagged[["logEUR", "lag 2"]].dropna(), maxlag=4)

"""
Granger Causality
number of lags (no zero) 1
ssr based F test:         F=4.5780  , p=0.0333  , df_denom=274, df_num=1
ssr based chi2 test:   chi2=4.6282  , p=0.0315  , df=1
likelihood ratio test: chi2=4.5899  , p=0.0322  , df=1
parameter F test:         F=4.5780  , p=0.0333  , df_denom=274, df_num=1
"""

eur_granger_lag3 = grangercausalitytests(eur_1year_lagged[["logEUR", "lag 3"]].dropna(), maxlag=4)

"""
Granger Causality
number of lags (no zero) 1
ssr based F test:         F=5.9640  , p=0.0152  , df_denom=273, df_num=1
ssr based chi2 test:   chi2=6.0295  , p=0.0141  , df=1
likelihood ratio test: chi2=5.9646  , p=0.0146  , df=1
parameter F test:         F=5.9640  , p=0.0152  , df_denom=273, df_num=1

Granger Causality
number of lags (no zero) 2
ssr based F test:         F=3.0560  , p=0.0487  , df_denom=270, df_num=2
ssr based chi2 test:   chi2=6.2252  , p=0.0445  , df=2
likelihood ratio test: chi2=6.1558  , p=0.0461  , df=2
parameter F test:         F=3.0560  , p=0.0487  , df_denom=270, df_num=2
"""

eur_granger_lag4 = grangercausalitytests(eur_1year_lagged[["logEUR", "lag 4"]].dropna(), maxlag=4)

"""
Granger Causality
number of lags (no zero) 1
ssr based F test:         F=7.0953  , p=0.0082  , df_denom=272, df_num=1
ssr based chi2 test:   chi2=7.1735  , p=0.0074  , df=1
likelihood ratio test: chi2=7.0816  , p=0.0078  , df=1
parameter F test:         F=7.0953  , p=0.0082  , df_denom=272, df_num=1
"""

eur_granger_lag5 = grangercausalitytests(eur_1year_lagged[["logEUR", "lag 5"]].dropna(), maxlag=4)

"""
Granger Causality
number of lags (no zero) 1
ssr based F test:         F=4.6650  , p=0.0317  , df_denom=271, df_num=1
ssr based chi2 test:   chi2=4.7167  , p=0.0299  , df=1
likelihood ratio test: chi2=4.6765  , p=0.0306  , df=1
parameter F test:         F=4.6650  , p=0.0317  , df_denom=271, df_num=1
"""

eur_granger_lag6 = grangercausalitytests(eur_1year_lagged[["logEUR", "lag 6"]].dropna(), maxlag=4)

"""
Granger Causality
number of lags (no zero) 1
ssr based F test:         F=5.2169  , p=0.0231  , df_denom=270, df_num=1
ssr based chi2 test:   chi2=5.2748  , p=0.0216  , df=1
likelihood ratio test: chi2=5.2245  , p=0.0223  , df=1
parameter F test:         F=5.2169  , p=0.0231  , df_denom=270, df_num=1

Granger Causality
number of lags (no zero) 2
ssr based F test:         F=4.7454  , p=0.0094  , df_denom=267, df_num=2
ssr based chi2 test:   chi2=9.6685  , p=0.0080  , df=2
likelihood ratio test: chi2=9.5006  , p=0.0086  , df=2
parameter F test:         F=4.7454  , p=0.0094  , df_denom=267, df_num=2
"""

# Although both "lag 3" and "lag 6" have significant values for both lag 1 and 2
# "lag 6" has lower p-values. There's could be a discussion that "lag 4" has lower
# values than "lag 6". I'd have to test models with these two lags to see what 
# happens 

# I've run other amount of lags for all of these, but mostly have p-values close
# to 1. 



# In[3.4]: As an experiment, running the EUR against USD:

usd_1year_granger = pd.read_csv("./datasets/wrangled/df_usd_1year.csv", float_precision="high", parse_dates=([0]))

df_eur_usd = pd.DataFrame({"eur": df_wg_eur_1year['logEUR'], "usd": usd_1year_granger["logUSD"]})

eur_usd_granger = grangercausalitytests(df_eur_usd[['eur', 'usd']].dropna(), maxlag=40)

"""
Granger Causality
number of lags (no zero) 28
ssr based F test:         F=1.1714  , p=0.2630  , df_denom=195, df_num=28
ssr based chi2 test:   chi2=42.3881 , p=0.0398  , df=28
likelihood ratio test: chi2=39.1784 , p=0.0781  , df=28
parameter F test:         F=1.1714  , p=0.2630  , df_denom=195, df_num=28

Granger Causality
number of lags (no zero) 29
ssr based F test:         F=1.2607  , p=0.1808  , df_denom=192, df_num=29
ssr based chi2 test:   chi2=47.7945 , p=0.0154  , df=29
likelihood ratio test: chi2=43.7501 , p=0.0388  , df=29
parameter F test:         F=1.2607  , p=0.1808  , df_denom=192, df_num=29

Granger Causality
number of lags (no zero) 30
ssr based F test:         F=1.3380  , p=0.1253  , df_denom=189, df_num=30
ssr based chi2 test:   chi2=53.0970 , p=0.0058  , df=30
likelihood ratio test: chi2=48.1480 , p=0.0192  , df=30
parameter F test:         F=1.3380  , p=0.1253  , df_denom=189, df_num=30

Granger Causality
number of lags (no zero) 31
ssr based F test:         F=1.3115  , p=0.1395  , df_denom=186, df_num=31
ssr based chi2 test:   chi2=54.4292 , p=0.0058  , df=31
likelihood ratio test: chi2=49.2262 , p=0.0200  , df=31
parameter F test:         F=1.3115  , p=0.1395  , df_denom=186, df_num=31

Granger Causality
number of lags (no zero) 32
ssr based F test:         F=1.2668  , p=0.1692  , df_denom=183, df_num=32
ssr based chi2 test:   chi2=54.9350 , p=0.0071  , df=32
likelihood ratio test: chi2=49.6222 , p=0.0242  , df=32
parameter F test:         F=1.2668  , p=0.1692  , df_denom=183, df_num=32
"""

# We can see here that some of the tests, specially the ones based on chi2 have
# very low p-values, but our F-tests never get below 0,12, which strongly indicates
# there's no addition of predictive power when adding the usd time series. Although
# we could make a case for the low p-values of the likelihood tests, specially
# at higher lags (28-32). For the purpose of this work it doesn't make sense because
# we're trying to predict values to a max of 2 weeks ahead, and the granger test
# shows us that the prediction power of the usd time series would work best in the
# long term and not in the short term, like we intend.

# In[3.5]: Testing VIF for EUR and USD:

eur_usd_constants = add_constant(df_eur_usd.dropna())

eur_usd_vif = pd.DataFrame()

eur_usd_vif['vif'] = [variance_inflation_factor(eur_usd_constants.values, i) for i in range(eur_usd_constants.shape[1])]

eur_usd_vif['variable'] = eur_usd_constants.columns

eur_usd_vif

"""
           vif variable
0  4262.280578    const
1     5.606104      eur
2     5.606104      usd
"""

# The low VIFs (< 10) indicate no multicolinearity between these two time series

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

"""
Lag: 1, P-value: 0.0004388260806742836
Lag: 2, P-value: 0.0030948639495272266
Lag: 3, P-value: 0.013025469160051002
Lag: 4, P-value: 0.04017990214602802

Lag: 1, P-value: 0.9448232487942886
Lag: 2, P-value: 0.9856853944964964
Lag: 3, P-value: 0.9428525632414568
Lag: 4, P-value: 0.9830950891931117

Lag: 1, P-value: 0.6446178408457518
Lag: 2, P-value: 0.9066326044178759
Lag: 3, P-value: 0.9078149102390983
Lag: 4, P-value: 0.9700173234563689

Lag: 1, P-value: 0.48895033921407816
Lag: 2, P-value: 0.7688838072846469
Lag: 3, P-value: 0.8780647953483983
Lag: 4, P-value: 0.9553963265831401

Lag: 1, P-value: 0.5235761299888084
Lag: 2, P-value: 0.4484979929950651
Lag: 3, P-value: 0.6288329617100512
Lag: 4, P-value: 0.8302239452202447
"""

# We can see that p-values for the lags across train/test sections is not consistent
# mostly losing any potential influencial power in later splits whem compared to the
# first one. So we can confidently assume that there's no causality between these two
# time series USD doesn't explain EUR or the relantionship between the two time series
# is sensitive to specific time windows or suffer influence from external factors

# In[4.0]: Save final dataset for testing ARIMA

eur_arima_1year = pd.DataFrame({
    "date": df_wg_eur_1year["dateTime"],
    "eur": df_wg_eur_1year["logEUR"],
    "diff": df_wg_eur_1year["diffLogEUR"],
    "lag4": eur_1year_lagged["lag 4"],
    "lag6": eur_1year_lagged["lag 6"]
    }).dropna()

# save to csv
eur_arima_1year.to_csv("./datasets/arima_ready/eur_arima_1year.csv", index=False)
