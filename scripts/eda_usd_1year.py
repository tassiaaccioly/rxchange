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

# Plot the 5 year dataset

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

# In[0.3]: Calculate Statistics for datasets

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

# In[0.5]: Plot the log dataset for visual assessment

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

# In[2.2]: Testing the lagged data for multicolinearity:

# Creating a new DataFrame with the lagged values:

usd_1year_lagged = pd.DataFrame({"diffLogUSD": df_wg_usd_1year["diffLogUSD"]})

usd_1year_lagged['lag 1'] = usd_1year_lagged["diffLogUSD"].shift(1)
usd_1year_lagged['lag 2'] = usd_1year_lagged["diffLogUSD"].shift(2)
usd_1year_lagged['lag 3'] = usd_1year_lagged["diffLogUSD"].shift(3)
usd_1year_lagged['lag 4'] = usd_1year_lagged["diffLogUSD"].shift(4)
usd_1year_lagged['lag 5'] = usd_1year_lagged["diffLogUSD"].shift(5)
usd_1year_lagged['lag 6'] = usd_1year_lagged["diffLogUSD"].shift(6)
usd_1year_lagged['lag 7'] = usd_1year_lagged["diffLogUSD"].shift(7)

usd_1year_lagged

usd_constants = add_constant(usd_1year_lagged.dropna())

usd_vif = pd.DataFrame()

usd_vif['vif'] = [variance_inflation_factor(usd_constants.values, i) for i in range(usd_constants.shape[1])]

usd_vif['variable'] = usd_constants.columns

usd_vif

"""
        vif    variable
0  1.039356       const
1  1.045368  diffLogUSD
2  1.040596       lag 1
3  1.067633       lag 2
4  1.058610       lag 3
5  1.057849       lag 4
6  1.038389       lag 5
7  1.038986       lag 6
8  1.068193       lag 7
"""

# We can see here there are no high VIFs, which means this is not necessarily a problem
# with multicolinearity between lags. Which could implicate we're using a high amount
# of lags, meaning we need to run the model again but limiting the amount of lags to less
# than the specified in the last model (which was 6)

# In[2.3]: Set a finite amount of lags to account for the second note on the OLS Regression:

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

# In[2.4]: Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test post differencing

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

# In[2.5]: Plotting ACF and PACF to determine correct number of lags for usd

plt.figure(figsize=(12,6))
plot_acf(df_wg_usd_1year['diffLogUSD'].dropna(), lags=13)
plt.show()

plt.figure(figsize=(12,6))
plot_pacf(df_wg_usd_1year['diffLogUSD'].dropna(), lags=13)
plt.show()

# Both tests seem to show normal stationary plots with no significant lags after zero.
# Also, no significant negative first lag;

# In[3.0]: Running Granger Causality tests to analyse the lags

# In[3.0]: Runnning the actual Causality test on the lagged data

usd_granger_1 = grangercausalitytests(usd_1year_lagged[["diffLogUSD", "lag 7"]].dropna(), maxlag=15)

usd_granger_1

"""
Granger Causality
number of lags (no zero) 1
ssr based F test:         F=7.9524  , p=0.0052  , df_denom=274, df_num=1
ssr based chi2 test:   chi2=8.0395  , p=0.0046  , df=1
likelihood ratio test: chi2=7.9250  , p=0.0049  , df=1
parameter F test:         F=7.9524  , p=0.0052  , df_denom=274, df_num=1
"""

# We have a definite confirmation on lag 1 for "lag 1" as you can see the p-value
# for all the tests are lower than the 0.05 critical value. Other than that, we do
# have a "lower" p-value on lag 6, but not enough (around 0.2), while other lags
# (2-30) are very close to p-value = 1, completely discarding them.
# This result suggests that, changes in the time series only affect values short-term,
# in this case, a day and then loose influence power, which means, this might be
# a good time series to use to predict values for usd for the next day, but only
# "lag 1". 

usd_granger_3 = grangercausalitytests(usd_1year_lagged[["diffLogUSD", "lag 3"]].dropna(), maxlag=15)

"""
Granger Causality
number of lags (no zero) 4
ssr based F test:         F=2.4935  , p=0.0434  , df_denom=264, df_num=4
ssr based chi2 test:   chi2=10.2761 , p=0.0360  , df=4
likelihood ratio test: chi2=10.0868 , p=0.0390  , df=4
parameter F test:         F=2.5425  , p=0.0402  , df_denom=264, df_num=4
"""

# "lag 3" seems to have meaningful p-values for lag 4.

usd_granger_4 = grangercausalitytests(usd_1year_lagged[["diffLogUSD", "lag 4"]].dropna(), maxlag=15)

"""
Granger Causality
number of lags (no zero) 3
ssr based F test:         F=3.3517  , p=0.0195  , df_denom=265, df_num=3
ssr based chi2 test:   chi2=10.3207 , p=0.0160  , df=3
likelihood ratio test: chi2=10.1297 , p=0.0175  , df=3
parameter F test:         F=3.3517  , p=0.0195  , df_denom=265, df_num=3
"""

# "lag 4" seems to have meaningful p-values for lag 3.

usd_granger_5 = grangercausalitytests(usd_1year_lagged[["diffLogUSD", "lag 5"]].dropna(), maxlag=15)

"""
Granger Causality
number of lags (no zero) 2
ssr based F test:         F=4.7181  , p=0.0097  , df_denom=267, df_num=2
ssr based chi2 test:   chi2=9.6129  , p=0.0082  , df=2
likelihood ratio test: chi2=9.4469  , p=0.0089  , df=2
parameter F test:         F=4.7181  , p=0.0097  , df_denom=267, df_num=2
"""

# "lag 5" seems to have meaningful p-values for lag 2.

usd_granger_6 = grangercausalitytests(usd_1year_lagged[["diffLogUSD", "lag 6"]].dropna(), maxlag=15)

"""
Granger Causality
number of lags (no zero) 1
ssr based F test:         F=10.5406 , p=0.0013  , df_denom=269, df_num=1
ssr based chi2 test:   chi2=10.6582 , p=0.0011  , df=1
likelihood ratio test: chi2=10.4546 , p=0.0012  , df=1
parameter F test:         F=10.5406 , p=0.0013  , df_denom=269, df_num=1

Granger Causality
number of lags (no zero) 2
ssr based F test:         F=4.5922  , p=0.0109  , df_denom=266, df_num=2
ssr based chi2 test:   chi2=9.3570  , p=0.0093  , df=2
likelihood ratio test: chi2=9.1991  , p=0.0101  , df=2
parameter F test:         F=4.5922  , p=0.0109  , df_denom=266, df_num=2

Granger Causality
number of lags (no zero) 3
ssr based F test:         F=3.0336  , p=0.0298  , df_denom=263, df_num=3
ssr based chi2 test:   chi2=9.3432  , p=0.0251  , df=3
likelihood ratio test: chi2=9.1851  , p=0.0269  , df=3
parameter F test:         F=3.0336  , p=0.0298  , df_denom=263, df_num=3
"""

# I ran "lag 7" but it had no meaningful p-value
# In this case, both for lowest p-value and more influence, we end up with "lag 6"
# It is good in the short term, but it can also help predict 3 days ahead

# In[3.1]: Cross-testing "diffLogUSD" and "lag 6"
# to make sure they have causality between them

usd_diff_lag6 = pd.DataFrame({"usd": df_wg_usd_1year["diffLogUSD"], "lag_6": usd_1year_lagged["lag 6"]})

tscv = TimeSeriesSplit(n_splits=5)

ct_usd_diff_lag6 = usd_diff_lag6.dropna()

for train_index, test_index in tscv.split(ct_usd_diff_lag6):
    train, test = ct_usd_diff_lag6.iloc[train_index], ct_usd_diff_lag6.iloc[test_index]

    X_train, y_train = train['lag_6'], train['usd']
    X_test, y_test = test['lag_6'], test['usd']

    granger_result = grangercausalitytests(train[['usd', 'lag_6']], maxlag=4, verbose=False)

    for lag, result in granger_result.items():
        f_test_pvalue = result[0]['ssr_ftest'][1]  # p-value from F-test
        print(f"Lag: {lag}, P-value: {f_test_pvalue}")

    print(f"TRAIN indices: {train_index}")
    print(f"TEST indices: {test_index}")

"""
Lag: 1, P-value: 0.03713425878383928
Lag: 2, P-value: 0.084291225205113
Lag: 3, P-value: 0.18924473982609968
Lag: 4, P-value: 0.3170353061542444

Lag: 1, P-value: 0.011896272230289547
Lag: 2, P-value: 0.007372819653804537
Lag: 3, P-value: 0.02459954634833422
Lag: 4, P-value: 0.07120700687269937

Lag: 1, P-value: 0.03357621187008599
Lag: 2, P-value: 0.042508109361530384
Lag: 3, P-value: 0.09365416832053626
Lag: 4, P-value: 0.16296626922795013

Lag: 1, P-value: 0.02249066181083836
Lag: 2, P-value: 0.03787921253070802
Lag: 3, P-value: 0.09357061012152754
Lag: 4, P-value: 0.1665320124084191

Lag: 1, P-value: 8.651104482384201e-05
Lag: 2, P-value: 0.0008640392575765902
Lag: 3, P-value: 0.0034883237613834717
Lag: 4, P-value: 0.008439935726456398
"""

# These values show that "lag 6" is a good predictor for both lag 1 and lag 2,
# loosing significance as we advance through lags. It is better in the short
# term than long term, although for lag 3 and 4, it shows that the p-value becomes
# significant as we approach the end of the data, which might mean, newer values
# (higher indices) have more impact on the predictions than older values.
# This indicated that "lag 6" and "diffLogUSD" have causality between them

# In[3.2]: Cross testing with VAR: