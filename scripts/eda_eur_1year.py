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
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import kpss


# In[0.2]: Import dataframes

df_wg_eur_1year = pd.read_csv("./datasets/wrangled/df_eur_1year.csv", float_precision="high", parse_dates=([0]))

df_wg_eur_1year.info()

df_wg_eur_5year = pd.read_csv("./datasets/wrangled/df_eur_5year.csv", float_precision="high", parse_dates=([0]))

df_wg_eur_5year

dt_format = "%d/%m/%Y"

# Plot the 5 year dataset

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

# In[0.3]: Calculate Statistics for datasets

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

# In[0.5]: Plot the log dataset for visual assessment

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
 
# In[1.4]: Differencing the data to achieve stationarity

df_wg_eur_1year['diffLogEUR'] = df_wg_eur_1year['logEUR'].diff()

df_wg_eur_1year

# In[1.5]: Plotting the differenced data for visual assessment:

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

# AIC, BIC and Loglike prove this model is even better than the second one.

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

# In[1.4]: Plotting ACF and PACF to determine correct number of lags for eur

plt.figure(figsize=(12,6))
plot_acf(df_wg_eur_1year['diffLogEUR'].dropna(), lags=13)
plt.show()

plt.figure(figsize=(12,6))
plot_pacf(df_wg_eur_1year['diffLogEUR'].dropna(), lags=13)
plt.show()

# Both tests seem to show normal stationary plots with no significant lags after zero.
# Also, no significant negative first lag;

