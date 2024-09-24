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

df_wg_usd_1year = pd.read_csv("./datasets/wrangled/df_usd_1year.csv", float_precision="high", parse_dates=([0]))

df_wg_usd_1year.info()

df_wg_usd_5year = pd.read_csv("./datasets/wrangled/df_usd_5year.csv", float_precision="high", parse_dates=([0]))

df_wg_usd_5year

dt_format = "%d/%m/%Y"

# In[0.3]: Plot the 5 year dataset

plt.figure(figsize=(15, 10))
sns.lineplot(x=df_wg_usd_5year["dateTime"], y=df_wg_usd_5year["usd"], color="limegreen", label="Câmbio USD")
plt.axhline(y=np.mean(df_wg_usd_5year["usd"]), color="black", linestyle="--", label="Média") # mean for usd
plt.axhline(y=np.max(df_wg_usd_5year["usd"]), color="magenta", label="Máxima") # max for usd
plt.axhline(y=np.min(df_wg_usd_5year["usd"]), color="magenta", linestyle="--", label="Mínima") # min for usd
plt.title(f'Cotação do Dóllar - Série histórica ({df_wg_usd_5year["dateTime"][0].strftime(dt_format)} - {df_wg_usd_5year["dateTime"][1740].strftime(dt_format)})', fontsize="18")
plt.yticks(np.arange(round(df_wg_usd_5year["usd"].min(), 1), round(df_wg_usd_5year["usd"].max() + 0.1, 1), 0.1), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b, %Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=90))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Câmbio USD ↔ BRL", fontsize="18")
plt.legend(fontsize=18, loc="upper right", bbox_to_anchor=(0.98, 0, 0, 0.32))
plt.show()

# In[0.4]: Calculate Statistics for datasets

var_usd_1year = np.var(df_wg_usd_1year['usd'])
var_usd_1year

# 0.021317213910427308

varlog_usd_1year = np.var(df_wg_usd_1year['logUSD'])
varlog_usd_1year

# 0.0008156800478685516

optimal_lags_usd_1year = 12*(len(df_wg_usd_1year['usd'])/100)**(1/4)
optimal_lags_usd_1year

# 15.522824731401618

# So optimal lags for this dataset = 15 or 16

# In[0.5]: Plot the original dataset for visual assessment

plt.figure(figsize=(15, 10))
sns.lineplot(x=df_wg_usd_1year["dateTime"], y=df_wg_usd_1year["usd"], color="limegreen", label="Câmbio USD")
plt.axhline(y=np.mean(df_wg_usd_1year["usd"]), color="black", linestyle="--", label="Média") # mean for usd
plt.axhline(y=np.max(df_wg_usd_1year["usd"]), color="magenta", label="Máxima") # max for usd
plt.axhline(y=np.min(df_wg_usd_1year["usd"]), color="magenta", linestyle="--", label="Mínima") # min for usd
plt.title(f'Cotação do Euro - Série histórica ({df_wg_usd_1year["dateTime"][0].strftime(dt_format)} - {df_wg_usd_1year["dateTime"][279].strftime(dt_format)})', fontsize="18")
plt.yticks(np.arange(round(df_wg_usd_1year["usd"].min(), 1), round(df_wg_usd_1year["usd"].max() + 0.1, 1), 0.1), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b, %Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=29))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Câmbio USD ↔ BRL", fontsize="18")
plt.legend(fontsize=18, loc="upper left", bbox_to_anchor=(0.02, 0, 0, 0.93))
plt.show()

# In[0.6]: Plot the original dataset for visual assessment

plt.figure(figsize=(15, 10))
sns.scatterplot(x=df_wg_usd_1year["dateTime"], y=df_wg_usd_1year["usd"], color="limegreen", label="Câmbio USD")
plt.axhline(y=np.mean(df_wg_usd_1year["usd"]), color="black", linestyle="--", label="Média") # mean for usd
plt.axhline(y=np.max(df_wg_usd_1year["usd"]), color="magenta", label="Máxima") # max for usd
plt.axhline(y=np.min(df_wg_usd_1year["usd"]), color="magenta", linestyle="--", label="Mínima") # min for usd
plt.title(f'Cotação do Euro - Série histórica ({df_wg_usd_1year["dateTime"][0].strftime(dt_format)} - {df_wg_usd_1year["dateTime"][279].strftime(dt_format)})', fontsize="18")
plt.yticks(np.arange(round(df_wg_usd_1year["usd"].min(), 1), round(df_wg_usd_1year["usd"].max() + 0.1, 1), 0.1), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b, %Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=29))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Câmbio USD ↔ BRL", fontsize="18")
plt.legend(fontsize=18, loc="upper left", bbox_to_anchor=(0.02, 0, 0, 0.93))
plt.show()

# In[0.7]: Plot the log dataset for visual assessment

plt.figure(figsize=(15, 10))
sns.lineplot(x=df_wg_usd_1year["dateTime"], y=df_wg_usd_1year["logUSD"], color="limegreen", label="log(Câmbio USD)")
plt.axhline(y=np.mean(df_wg_usd_1year["logUSD"]), color="black", linestyle="--", label="Média") # mean for usd
plt.axhline(y=np.max(df_wg_usd_1year["logUSD"]), color="magenta", label="Máxima") # max for usd
plt.axhline(y=np.min(df_wg_usd_1year["logUSD"]), color="magenta", linestyle="--", label="Mínima") # min for usd
plt.title(f'Cotação do Euro - Série histórica ({df_wg_usd_1year["dateTime"][0].strftime(dt_format)} - {df_wg_usd_1year["dateTime"][279].strftime(dt_format)})', fontsize="18")
plt.yticks(np.arange(round(df_wg_usd_1year["logUSD"].min(), 2), round(df_wg_usd_1year["logUSD"].max() + 0.01, 2), 0.01), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b, %Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=29))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Câmbio log(USD ↔ BRL)", fontsize="18")
plt.legend(fontsize=18, loc="upper left", bbox_to_anchor=(0.02, 0, 0, 0.93))
plt.show()


# In[1.0]: Defining Stationarity

# In[1.1]: Augmented Dickey-Fuller Test with usd and logUSD

usd_1year_adf_ols = adfuller(df_wg_usd_1year['usd'], maxlag=None, autolag="AIC", store=True, regresults=True)[-1].resols.summary()
usd_1year_adf_ols

"""
                            OLS Regression Results
==============================================================================
Dep. Variable:                      y   R-squared:                       0.076
Model:                            OLS   Adj. R-squared:                  0.048
Method:                 Least Squares   F-statistic:                     2.704
Date:                Sun, 22 Sep 2024   Prob (F-statistic):            0.00708
Time:                        13:13:50   Log-Likelihood:                 603.13
No. Observations:                 272   AIC:                            -1188.
Df Residuals:                     263   BIC:                            -1156.
Df Model:                           8
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1             0.0164      0.013      1.304      0.193      -0.008       0.041
x2            -0.0636      0.063     -1.014      0.312      -0.187       0.060
x3            -0.1839      0.062     -2.943      0.004      -0.307      -0.061
x4            -0.0453      0.064     -0.710      0.479      -0.171       0.080
x5            -0.0456      0.064     -0.718      0.473      -0.171       0.079
x6             0.0387      0.063      0.609      0.543      -0.086       0.164
x7             0.0160      0.063      0.255      0.799      -0.108       0.140
x8            -0.1876      0.063     -2.996      0.003      -0.311      -0.064
const         -0.0799      0.063     -1.266      0.207      -0.204       0.044
==============================================================================
Omnibus:                       25.549   Durbin-Watson:                   1.978
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               56.371
Skew:                           0.459   Prob(JB):                     5.74e-13
Kurtosis:                       5.033   Cond. No.                         278.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

# Doing the same for the log database

logusd_1year_adf_ols = adfuller(df_wg_usd_1year['logUSD'], maxlag=None, autolag="AIC", store=True, regresults=True)[-1].resols.summary()
logusd_1year_adf_ols

"""
                            OLS Regression Results
==============================================================================
Dep. Variable:                      y   R-squared:                       0.076
Model:                            OLS   Adj. R-squared:                  0.047
Method:                 Least Squares   F-statistic:                     2.686
Date:                Sun, 22 Sep 2024   Prob (F-statistic):            0.00744
Time:                        13:14:51   Log-Likelihood:                 1045.2
No. Observations:                 272   AIC:                            -2072.
Df Residuals:                     263   BIC:                            -2040.
Df Model:                           8
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1             0.0147      0.013      1.173      0.242      -0.010       0.039
x2            -0.0619      0.063     -0.988      0.324      -0.185       0.061
x3            -0.1837      0.062     -2.948      0.003      -0.306      -0.061
x4            -0.0483      0.064     -0.759      0.449      -0.174       0.077
x5            -0.0399      0.063     -0.630      0.529      -0.165       0.085
x6             0.0370      0.063      0.586      0.558      -0.087       0.161
x7             0.0177      0.062      0.284      0.777      -0.105       0.141
x8            -0.1856      0.062     -2.978      0.003      -0.308      -0.063
const         -0.0233      0.020     -1.151      0.251      -0.063       0.017
==============================================================================
Omnibus:                       22.965   Durbin-Watson:                   1.979
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               48.996
Skew:                           0.418   Prob(JB):                     2.29e-11
Kurtosis:                       4.903   Cond. No.                         476.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

# We can notice that in the second model, both AIC and BIC are lower than the first one
# and also that the loglike is higher on the second model than the first one, proving
# that a logarthmic version of the data is a better fit for the model so far.

logusd_1year_adf = adfuller(df_wg_usd_1year["logUSD"], maxlag=None, autolag="AIC")
logusd_1year_adf

# p-value is > 0.05 (0.995), DO NOT REJECT the null hypothesis and suggests the presence of a unit root (NON-STATIONARY)
# adf stats > 10% (1.17), DO NOT REJECT the null hypothesis and suggests the presence of a unit root (NON-STATIONARY)
# lags = 7 good amount of lags

"""
(1.1732167812125203,
 0.99579551071698,
 7,
 272,
 {'1%': -3.4546223782586534,
  '5%': -2.8722253212300277,
  '10%': -2.5724638500216264},
 -2014.0704788304256)
"""

# In[1.2]: Running KPSS test to determine stationarity for usd

logusd_1year_kpss = kpss(df_wg_usd_1year['logUSD'], regression="c", nlags="auto")
logusd_1year_kpss

# p-value < 0.05 (0.01) REJECTS the null hypothesis, suggesting data is NON-STATIONARY
# kpss stat of 1.52 is > than 1% (0.739), REJECTS the null hypothesis, suggesting data is NON-STATIONARY
# lags = 10 good amount of lags

"""
(1.5244497884900459,
 0.01,
 10,
 {'10%': 0.347, '5%': 0.463, '2.5%': 0.574, '1%': 0.739})
"""

# In[1.3]: Plotting ACF and PACF to examine autocorrelations in data:

plt.figure(figsize=(12,6))
plot_acf(df_wg_usd_1year['logUSD'].dropna(), lags=15)
plt.show()

plt.figure(figsize=(12,6))
plot_pacf(df_wg_usd_1year['logUSD'].dropna(), lags=15)
plt.show()

# PACF plot shows an AR(2) order for the dataset showing a high statistical significance spike
# at the first lag in PACF. ACF shows slow decay towards 0 -> exponentially decaying or sinusoidal

# In[1.4]: Defining the order of differencing we need:

plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120}) 

# Original Series
fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot(df_wg_usd_1year["logUSD"]);
ax1.set_title('Série Original log(USD)');
ax1.axes.xaxis.set_visible(False)

# 1st Differencing
ax2.plot(df_wg_usd_1year["logUSD"].diff(), color="limegreen");
ax2.set_title('1ª Ordem de Diferenciação');
ax2.axes.xaxis.set_visible(False)

# 2nd Differencing
ax3.plot(df_wg_usd_1year["logUSD"].diff().diff());
ax3.set_title('2ª Ordem de Diferenciação') 
plt.show()

# Plotting the ACF for each order

plt.rcParams.update({'figure.figsize':(11,10), 'figure.dpi':120})

fig, (ax1, ax2, ax3) = plt.subplots(3)
plot_acf(df_wg_usd_1year["logUSD"], ax=ax1)
plot_acf(df_wg_usd_1year["logUSD"].diff().dropna(), ax=ax2)
plot_acf(df_wg_usd_1year["logUSD"].diff().diff().dropna(), ax=ax3)

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
    
plot_pacf(df_wg_usd_1year["logUSD"].diff().dropna())
plt.show()

# Plotting ACF and PACF together:

plt.rcParams.update({'figure.figsize':(11,8), 'figure.dpi':120})

fig, (ax1, ax2) = plt.subplots(2)
plot_acf(df_wg_usd_1year["logUSD"].diff().dropna(), ax=ax1)
plot_pacf(df_wg_usd_1year["logUSD"].diff().dropna(), ax=ax2)

# These plots show a sharp cut off at ACF lag 2, which indicates sligh overdifferencing
# and also an MA parameter of 2. The stationarized series display an "MA signature"
# Meaning we can explain the autocorrelation pattern my adding MA terms rather than
# AR terms. PACF is related to AR (orders while ACF is related to MA (lags of the forecast
# errors). We can see from the PACF plot that after the first lag we don't have
# any other positive lags outside of the significance limit, so we 

# We can see a pretty good visually STATIONARY plot on the first differenciation,
# so we'll be going on with a single differenciation and we'll check how that looks
# with ACF and PACF

# In[1.4]: Differencing the data to achieve stationarity

df_wg_usd_1year['diffLogUSD'] = df_wg_usd_1year['logUSD'].diff()

df_wg_usd_1year

# In[1.5]: Plotting the differenced data for visual assessment:

plt.figure(figsize=(15, 10))
sns.lineplot(x=df_wg_usd_1year["dateTime"], y=df_wg_usd_1year["diffLogUSD"], color="limegreen", label="Log(USD) - Diff")
plt.axhline(y=np.mean(df_wg_usd_1year["diffLogUSD"]), color="black", linestyle="--", label="Média") # mean for usd
plt.title(f'Log do Dólar Diferenciado - Série histórica ({df_wg_usd_1year["dateTime"][0].strftime(dt_format)} - {df_wg_usd_1year["dateTime"][279].strftime(dt_format)})', fontsize="18")
plt.yticks(np.arange(round(df_wg_usd_1year["diffLogUSD"].min(), 3), round(df_wg_usd_1year["diffLogUSD"].max() + 0.002, 3), 0.002), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b, %Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=29))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Log(Câmbio USD ↔ BRL) Diferenciado ", fontsize="18")
plt.legend(fontsize=18, loc="upper center")
plt.show()

# In[2.0]: Defining Stationarity again

optimal_lags_usd_1year_diff = 12*(len(df_wg_usd_1year['diffLogUSD'].dropna())/100)**(1/4)
optimal_lags_usd_1year_diff

# 15.508946465645465

# So optimal lags for this dataset = 15 or 16


# In[2.1]: Augmented Dickey-Fuller Test with diffLogUSD

logusd_1year_diff_adf = adfuller(df_wg_usd_1year["diffLogUSD"].dropna(), maxlag=None, autolag="AIC")
logusd_1year_diff_adf

# p-value is << 0.05 (1.668561287686211e-10), REJECTS the null hypothesis and suggests the absence of a unit root (STATIONARY)
# adf stats << 1% (-7), REJECTS the null hypothesis and suggests the absence of a unit root (STATIONARY)
# lags = 1 good amount of lags

"""
(-7.262375073622774,
 1.668561287686211e-10,
 6,
 272,
 {'1%': -3.4546223782586534,
  '5%': -2.8722253212300277,
  '10%': -2.5724638500216264},
 -2005.6481155638012)
"""

logusd_1year_diff_adf_ols = adfuller(df_wg_usd_1year['diffLogUSD'].dropna(), maxlag=None, autolag="AIC", store=True, regresults=True)[-1].resols.summary()
logusd_1year_diff_adf_ols

"""
                            OLS Regression Results
==============================================================================
Dep. Variable:                      y   R-squared:                       0.545
Model:                            OLS   Adj. R-squared:                  0.533
Method:                 Least Squares   F-statistic:                     45.18
Date:                Sun, 22 Sep 2024   Prob (F-statistic):           9.89e-42
Time:                        13:23:47   Log-Likelihood:                 1044.5
No. Observations:                 272   AIC:                            -2073.
Df Residuals:                     264   BIC:                            -2044.
Df Model:                           7
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1            -1.3583      0.187     -7.262      0.000      -1.727      -0.990
x2             0.3122      0.173      1.804      0.072      -0.029       0.653
x3             0.1428      0.158      0.906      0.366      -0.168       0.453
x4             0.1113      0.138      0.807      0.420      -0.160       0.383
x5             0.0876      0.115      0.761      0.448      -0.139       0.314
x6             0.1404      0.089      1.584      0.114      -0.034       0.315
x7             0.1725      0.061      2.811      0.005       0.052       0.293
const          0.0005      0.000      1.383      0.168      -0.000       0.001
==============================================================================
Omnibus:                       24.178   Durbin-Watson:                   1.971
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               50.366
Skew:                           0.452   Prob(JB):                     1.16e-11
Kurtosis:                       4.904   Cond. No.                     1.05e+03
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 1.05e+03. This might indicate that there are
strong multicollinearity or other numerical problems.
"""

# The note[2] in the OLS Regression is worrisome, but we can mitigate that by running
# these tests again but setting a finite amount lags. This might increase the test
# stat and decrease the value of the p-value, making it seem more over-fitted, but
# this will leave it like the eur dataset basically. Let's hope the kpss also shows
# a "normal" result.
# This run also shows no statistical improvement  in AIC and BIC from the last OLS
# Regression, and it actually shows the Loglike is slighly lower in this one.


# In[2.2]: Set a finite amount of lags to account for the second note on the OLS Regression:

logusd_1year_diff_adf = adfuller(df_wg_usd_1year["diffLogUSD"].dropna(), maxlag=5, autolag="AIC")
logusd_1year_diff_adf

# p-value is << 0.05 (1.668561287686211e-10), REJECTS the null hypothesis and suggests the absence of a unit root (STATIONARY)
# adf stats << 1% (-7), REJECTS the null hypothesis and suggests the absence of a unit root (STATIONARY)
# lags = 1 good amount of lags

"""
(-13.808097123348794,
 8.329065910491463e-26,
 1,
 277,
 {'1%': -3.4541800885158525,
  '5%': -2.872031361137725,
  '10%': -2.5723603999791473},
 -2080.777374508495)
"""

logusd_1year_diff_adf_ols = adfuller(df_wg_usd_1year['diffLogUSD'].dropna(), maxlag=5, autolag="AIC", store=True, regresults=True)[-1].resols.summary()
logusd_1year_diff_adf_ols

"""
                            OLS Regression Results
==============================================================================
Dep. Variable:                      y   R-squared:                       0.523
Model:                            OLS   Adj. R-squared:                  0.519
Method:                 Least Squares   F-statistic:                     150.1
Date:                Sun, 22 Sep 2024   Prob (F-statistic):           9.41e-45
Time:                        13:33:09   Log-Likelihood:                 1057.0
No. Observations:                 277   AIC:                            -2108.
Df Residuals:                     274   BIC:                            -2097.
Df Model:                           2
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1            -1.2071      0.087    -13.808      0.000      -1.379      -1.035
x2             0.1711      0.061      2.820      0.005       0.052       0.291
const          0.0005      0.000      1.464      0.144      -0.000       0.001
==============================================================================
Omnibus:                       22.028   Durbin-Watson:                   1.998
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               45.917
Skew:                           0.401   Prob(JB):                     1.07e-10
Kurtosis:                       4.826   Cond. No.                         311.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

# This time it gave us a better result for the OLS, it also gave better results for the
# AIC, BIC and Loglike stats, making this a better model than before.

# In[2.3]: Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test post differencing

logusd_1year_diff_kpss = kpss(df_wg_usd_1year['diffLogUSD'].dropna(), regression="c", nlags="auto")
logusd_1year_diff_kpss

# p-value > 0.05 (0.08) DO NOT REJECT the null hypothesis, suggesting data is STATIONARY
# kpss stat of 0.37 is < 10% (0.347), DO NOT REJECT the null hypothesis, suggesting data is STATIONARY
# lags = 10 ok amount of lags for this point in time

"""
(0.37871530258782904,
 0.08632961095352197,
 10,
 {'10%': 0.347, '5%': 0.463, '2.5%': 0.574, '1%': 0.739})
"""

# The adf test shows similar results to the ones for EURO, which could mean a bad time.
# KPSS though shows almost better (?) results than the EURO one, but still validates stationarity
# although it's veeery close to not being stationary as both the p-value and kpss stats are
# very close to the limit. This also has a high number of lags, but it's ok.

# In[2.4]: Plotting ACF and PACF to determine correct number of lags for usd

plt.figure(figsize=(12,6))
plot_acf(df_wg_usd_1year['diffLogUSD'].dropna(), lags=13)
plt.show()

plt.figure(figsize=(12,6))
plot_pacf(df_wg_usd_1year['diffLogUSD'].dropna(), lags=13)
plt.show()

# Both tests seem to show normal stationary plots with no significant lags after zero.
# Also, no significant negative first lag;

# In[2.5]: Testing the lagged data for multicolinearity:

# Creating a new DataFrame with the lagged values:

usd_1year_lagged = pd.DataFrame({"logUSD": df_wg_usd_1year["logUSD"]})

usd_1year_lagged['lag 1'] = usd_1year_lagged["logUSD"].shift(1)
usd_1year_lagged['lag 2'] = usd_1year_lagged["logUSD"].shift(2)
usd_1year_lagged['lag 3'] = usd_1year_lagged["logUSD"].shift(3)
usd_1year_lagged['lag 4'] = usd_1year_lagged["logUSD"].shift(4)
usd_1year_lagged['lag 5'] = usd_1year_lagged["logUSD"].shift(5)
usd_1year_lagged['lag 6'] = usd_1year_lagged["logUSD"].shift(6)
usd_1year_lagged['lag 7'] = usd_1year_lagged["logUSD"].shift(7)

usd_1year_lagged

usd_constants = add_constant(usd_1year_lagged.dropna())

usd_vif = pd.DataFrame()

usd_vif['vif'] = [variance_inflation_factor(usd_constants.values, i) for i in range(usd_constants.shape[1])]

usd_vif['variable'] = usd_constants.columns

usd_vif

"""
           vif variable
0  3902.782455    const
1    29.971152   logUSD
2    54.509840    lag 1
3    53.039979    lag 2
4    53.358767    lag 3
5    51.838848    lag 4
6    50.262169    lag 5
7    48.502724    lag 6
8    25.482231    lag 7
"""

# We can see here a lot of high VIFs, but this is expected, since these values
# como from just lagged datasets they are bound to have multicolinearity, which
# is not necessarily a problem, we just need to be cautions to the amount of
# lagged exogs we add at the end

# In[3.0]: Running Granger Causality tests to analyse the lags

# In[3.0]: Runnning the actual Causality test on the lagged data

usd_granger_1 = grangercausalitytests(usd_1year_lagged[["logUSD", "lag 1"]].dropna(), maxlag=4)

usd_granger_1

"""
Granger Causality
number of lags (no zero) 2
ssr based F test:         F=4.1261  , p=0.0172  , df_denom=273, df_num=2
ssr based chi2 test:   chi2=8.3731  , p=0.0152  , df=2
likelihood ratio test: chi2=8.2490  , p=0.0162  , df=2
parameter F test:         F=4.2447  , p=0.0153  , df_denom=273, df_num=2
"""

usd_granger_2 = grangercausalitytests(usd_1year_lagged[["logUSD", "lag 2"]].dropna(), maxlag=4)

"""
Granger Causality
number of lags (no zero) 1
ssr based F test:         F=5.8819  , p=0.0159  , df_denom=274, df_num=1
ssr based chi2 test:   chi2=5.9463  , p=0.0147  , df=1
likelihood ratio test: chi2=5.8834  , p=0.0153  , df=1
parameter F test:         F=5.8819  , p=0.0159  , df_denom=274, df_num=1

Granger Causality
number of lags (no zero) 2
ssr based F test:         F=4.4626  , p=0.0124  , df_denom=271, df_num=2
ssr based chi2 test:   chi2=9.0900  , p=0.0106  , df=2
likelihood ratio test: chi2=8.9435  , p=0.0114  , df=2
parameter F test:         F=4.4626  , p=0.0124  , df_denom=271, df_num=2
"""

usd_granger_4 = grangercausalitytests(usd_1year_lagged[["logUSD", "lag 4"]].dropna(), maxlag=4)

"""
Granger Causality
number of lags (no zero) 1
ssr based F test:         F=5.0776  , p=0.0250  , df_denom=272, df_num=1
ssr based chi2 test:   chi2=5.1337  , p=0.0235  , df=1
likelihood ratio test: chi2=5.0863  , p=0.0241  , df=1
parameter F test:         F=5.0776  , p=0.0250  , df_denom=272, df_num=1

Granger Causality
number of lags (no zero) 2
ssr based F test:         F=3.0976  , p=0.0468  , df_denom=269, df_num=2
ssr based chi2 test:   chi2=6.3104  , p=0.0426  , df=2
likelihood ratio test: chi2=6.2388  , p=0.0442  , df=2
parameter F test:         F=3.0976  , p=0.0468  , df_denom=269, df_num=2

Granger Causality
number of lags (no zero) 4
ssr based F test:         F=2.6949  , p=0.0314  , df_denom=263, df_num=4
ssr based chi2 test:   chi2=11.1484 , p=0.0249  , df=4
likelihood ratio test: chi2=10.9260 , p=0.0274  , df=4
parameter F test:         F=2.6949  , p=0.0314  , df_denom=263, df_num=4
"""

usd_granger_5 = grangercausalitytests(usd_1year_lagged[["logUSD", "lag 5"]].dropna(), maxlag=4)

"""
Granger Causality
number of lags (no zero) 3
ssr based F test:         F=3.4053  , p=0.0182  , df_denom=265, df_num=3
ssr based chi2 test:   chi2=10.4857 , p=0.0149  , df=3
likelihood ratio test: chi2=10.2886 , p=0.0163  , df=3
parameter F test:         F=3.4053  , p=0.0182  , df_denom=265, df_num=3

Granger Causality
number of lags (no zero) 4
ssr based F test:         F=2.4862  , p=0.0440  , df_denom=262, df_num=4
ssr based chi2 test:   chi2=10.2865 , p=0.0359  , df=4
likelihood ratio test: chi2=10.0960 , p=0.0388  , df=4
parameter F test:         F=2.4862  , p=0.0440  , df_denom=262, df_num=4
"""

usd_granger_6 = grangercausalitytests(usd_1year_lagged[["logUSD", "lag 6"]].dropna(), maxlag=4)

"""
Granger Causality
number of lags (no zero) 2
ssr based F test:         F=6.3854  , p=0.0020  , df_denom=267, df_num=2
ssr based chi2 test:   chi2=13.0100 , p=0.0015  , df=2
likelihood ratio test: chi2=12.7084 , p=0.0017  , df=2
parameter F test:         F=6.3854  , p=0.0020  , df_denom=267, df_num=2

Granger Causality
number of lags (no zero) 3
ssr based F test:         F=3.2917  , p=0.0212  , df_denom=264, df_num=3
ssr based chi2 test:   chi2=10.1368 , p=0.0174  , df=3
likelihood ratio test: chi2=9.9518  , p=0.0190  , df=3
parameter F test:         F=3.2917  , p=0.0212  , df_denom=264, df_num=3

Granger Causality
number of lags (no zero) 4
ssr based F test:         F=2.4436  , p=0.0471  , df_denom=261, df_num=4
ssr based chi2 test:   chi2=10.1115 , p=0.0386  , df=4
likelihood ratio test: chi2=9.9267  , p=0.0417  , df=4
parameter F test:         F=2.4436  , p=0.0471  , df_denom=261, df_num=4
"""

# The one with the lowest values is lag 6. We'll use that.

# In[3.1]: Cross-testing "diffLogUSD" and "lag 6"
# to make sure they have causality between them

usd_diff_lag6 = pd.DataFrame({"usd": df_wg_usd_1year["logUSD"], "lag_6": usd_1year_lagged["lag 6"]})

tscv = TimeSeriesSplit(n_splits=5)

ct_usd_diff_lag6 = usd_diff_lag6.dropna()

for train_index, test_index in tscv.split(ct_usd_diff_lag6):
    train, test = ct_usd_diff_lag6.iloc[train_index], ct_usd_diff_lag6.iloc[test_index]

    X_train, y_train = train['lag_6'], train['usd']
    X_test, y_test = test['lag_6'], test['usd']

    granger_result = grangercausalitytests(train[['usd', 'lag_6']], maxlag=6, verbose=False)

    for lag, result in granger_result.items():
        f_test_pvalue = result[0]['ssr_ftest'][1]  # p-value from F-test
        print(f"Lag: {lag}, P-value: {f_test_pvalue}")

    print(f"TRAIN indices: {train_index}")
    print(f"TEST indices: {test_index}")

"""
Lag: 1, P-value: 0.5557644518769467
Lag: 2, P-value: 0.14360387036477595
Lag: 3, P-value: 0.19318909337034024
Lag: 4, P-value: 0.28115411753441255

Lag: 1, P-value: 0.9335885155472131
Lag: 2, P-value: 0.06822797137298005
Lag: 3, P-value: 0.018103686248593074
Lag: 4, P-value: 0.03802228466927357

Lag: 1, P-value: 0.9374524960366032
Lag: 2, P-value: 0.16715507302834273
Lag: 3, P-value: 0.08952825409920961
Lag: 4, P-value: 0.12781108755623943

Lag: 1, P-value: 0.6752417508708108
Lag: 2, P-value: 0.10431048332417303
Lag: 3, P-value: 0.09375084402346995
Lag: 4, P-value: 0.14320787279129513

Lag: 1, P-value: 0.8946013951852878
Lag: 2, P-value: 0.0007264486358554861
Lag: 3, P-value: 0.0038570309030882054
Lag: 4, P-value: 0.008418990918427283
"""

# These show that lag 6 actually is not good at predicting values for this dataframe
# We might still take it to test in the arima model, but it seems like it won't
# give a good outcome.

# In[3.2]: Testing Granger Causality with "diffLogUSD" and "eur"

df_wg_eur_1year = pd.read_csv("./datasets/wrangled/df_eur_1year.csv", float_precision="high", parse_dates=([0]))

df_usd_eur = pd.DataFrame({"usd": df_wg_usd_1year['logUSD'], "eur": df_wg_eur_1year["logEUR"]})

usd_eur_granger = grangercausalitytests(df_usd_eur[['usd', 'eur']].dropna(), maxlag=40)


# It looks like Euro could be a great predictor for USD. There are significant
# predictor values from lag 7 to lag 37. This sounds insane. I'll check these
# in the ARIMA models and get the best by using AIC and BIC.

# In[2.5]: Testing the eur and usd data for multicolinearity:

usd_eur_vif_test = pd.DataFrame({"logUSD": df_wg_usd_1year["logUSD"], "logEUR": df_wg_eur_1year["logEUR"]})

usd_eur_vif_test

usd_constants = add_constant(usd_eur_vif_test.dropna())

usd_eur_vif = pd.DataFrame()

usd_eur_vif['vif'] = [variance_inflation_factor(usd_constants.values, i) for i in range(usd_constants.shape[1])]

usd_eur_vif['variable'] = usd_constants.columns

usd_eur_vif

"""
           vif variable
0  4262.280578    const
1     5.606104   logUSD
2     5.606104   logEUR
"""

# Although there is not a "high" VIF value (> 10) these could still pose a problem
# for multicolinearity? I need to investigate.

# In[4.0]: Save final dataset for testing ARIMA

usd_arima_1year = pd.DataFrame({
    "date": df_wg_usd_1year["dateTime"],
    "usd": df_wg_usd_1year["logUSD"],
    "diff": df_wg_usd_1year["diffLogUSD"],
    "lag": usd_1year_lagged["lag 6"],
    "eur": df_wg_eur_1year["logEUR"],
    "eurDiff": df_wg_eur_1year["logEUR"].diff()
    }).dropna()

# save to csv
usd_arima_1year.to_csv("./datasets/arima_ready/usd_arima_1year.csv", index=False)
