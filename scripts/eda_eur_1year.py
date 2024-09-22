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
sns.lineplot(x=df_wg_eur_5year["dateTime"], y=df_wg_eur_5year["eur"], color="limegreen", label="EUR (€)")
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
plt.ylabel("Câmbio EUR → BRL", fontsize="18")
plt.legend(fontsize=18, loc="upper left", bbox_to_anchor=(0.02, 0, 0, 0.93))
plt.show()

# In[0.3]: Calculate Statistics for datasets

var_eur_1year = np.var(df_wg_eur_1year['eur'])
var_eur_1year

#0.020617129582265307

varlog_eur_1year = np.var(df_wg_eur_1year['logEUR'])
varlog_eur_1year

#0.0006763790454904603

# In[0.5]: Plot the original dataset for visual assessment

plt.figure(figsize=(15, 10))
sns.lineplot(x=df_wg_eur_1year["dateTime"], y=df_wg_eur_1year["eur"], color="limegreen", label="EUR (€)")
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
plt.ylabel("Câmbio EUR → BRL", fontsize="18")
plt.legend(fontsize=18, loc="upper left", bbox_to_anchor=(0.02, 0, 0, 0.93))
plt.show()

# In[0.5]: Plot the log dataset for visual assessment

plt.figure(figsize=(15, 10))
sns.lineplot(x=df_wg_eur_1year["dateTime"], y=df_wg_eur_1year["logEUR"], color="limegreen", label="EUR → BRL")
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
plt.ylabel("Câmbio log(EUR → BRL)", fontsize="18")
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
# and also that the loglike is higher on the second model than the first one.

# Getting values from the second model:
    
logeur_1year_adf = adfuller(df_wg_eur_1year["logEUR"], maxlag=None, autolag="AIC")
logeur_1year_adf

# p-value is > 0.05 (0.99), DO NOT REJECT the null hypothesis and suggests the presence of a unit root (NON-STATIONARY)
# adf stats > 10% (1.20), DO NOT REJECT the null hypothesis and suggests the presence of a unit root (NON-STATIONARY)
# lags = 7

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
# lags = 10

"""
(2.004541680328264,
 0.01,
 10,
 {'10%': 0.347, '5%': 0.463, '2.5%': 0.574, '1%': 0.739})
"""

# In[1.3]: Differencing the data to achieve stationarity

df_wg_eur_1year['diffLogEUR'] = df_wg_eur_1year['logEUR'].diff()

df_wg_eur_1year

# In[2.0]: Defining Stationarity again

# In[2.1]: Augmented Dickey-Fuller Test with diffLogEUR

logeur_1year_diff_adf = adfuller(df_wg_eur_1year["diffLogEUR"].dropna(), maxlag=None, autolag="AIC")
logeur_1year_diff_adf

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

eur_1year_diff_adf_ols = adfuller(df_wg_eur_1year['diffEUR'].dropna(), maxlag=None, autolag="AIC", store=True, regresults=True)[-1].resols.summary()
eur_1year_diff_adf_ols

# In[1.4]: Plotting ACF and PACF to determine correct number of lags for eur

plt.figure(figsize=(12,6))
plot_acf(df_wg_eur_1year['logEUR'], lags=50)
plt.show()

plt.figure(figsize=(12,6))
plot_pacf(df_wg_eur_1year['logEUR'], lags=16)
plt.show()

