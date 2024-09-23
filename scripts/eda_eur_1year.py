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

eur_1year_lagged = pd.DataFrame({"diffLogEUR": df_wg_eur_1year["diffLogEUR"]})

eur_1year_lagged['lag 1'] = eur_1year_lagged["diffLogEUR"].shift(1)
eur_1year_lagged['lag 2'] = eur_1year_lagged["diffLogEUR"].shift(2)
eur_1year_lagged['lag 3'] = eur_1year_lagged["diffLogEUR"].shift(3)
eur_1year_lagged['lag 4'] = eur_1year_lagged["diffLogEUR"].shift(4)
eur_1year_lagged['lag 5'] = eur_1year_lagged["diffLogEUR"].shift(5)
eur_1year_lagged['lag 6'] = eur_1year_lagged["diffLogEUR"].shift(6)

eur_1year_lagged

# Running only one lag as this was what adf showed was optimal

# In[3.2]: Running Multicolinearity tests just in case:

eur_constants = add_constant(eur_1year_lagged.dropna())

eur_vif = pd.DataFrame()

eur_vif['vif'] = [variance_inflation_factor(eur_constants.values, i) for i in range(eur_constants.shape[1])]

eur_vif['variable'] = eur_constants.columns

eur_vif

"""
        vif    variable
0  1.013739       const
1  1.004497  diffLogEUR
2  1.004497       lag 1
"""

# We can see the low VIFs indicate no multicolinearity for this dataframe

# In[3.3]: Runnning the actual Causality test on the lagged data

eur_granger = grangercausalitytests(eur_1year_lagged[["diffLogEUR", "lag 1"]].dropna(), maxlag=4)

eur_granger

"""
Granger Causality
number of lags (no zero) 1
ssr based F test:         F=3.5121  , p=0.0620  , df_denom=274, df_num=1
ssr based chi2 test:   chi2=3.5505  , p=0.0595  , df=1
likelihood ratio test: chi2=3.5279  , p=0.0603  , df=1
parameter F test:         F=3.5121  , p=0.0620  , df_denom=274, df_num=1

Granger Causality
number of lags (no zero) 2
ssr based F test:         F=0.7026  , p=0.4962  , df_denom=272, df_num=2
ssr based chi2 test:   chi2=1.4258  , p=0.4902  , df=2
likelihood ratio test: chi2=1.4221  , p=0.4911  , df=2
parameter F test:         F=2.4611  , p=0.0872  , df_denom=272, df_num=2

Granger Causality
number of lags (no zero) 3
ssr based F test:         F=0.4023  , p=0.7515  , df_denom=270, df_num=3
ssr based chi2 test:   chi2=1.2292  , p=0.7460  , df=3
likelihood ratio test: chi2=1.2264  , p=0.7467  , df=3
parameter F test:         F=2.0329  , p=0.1096  , df_denom=270, df_num=3

Granger Causality
number of lags (no zero) 4
ssr based F test:         F=0.0096  , p=0.9998  , df_denom=268, df_num=4
ssr based chi2 test:   chi2=0.0393  , p=0.9998  , df=4
likelihood ratio test: chi2=0.0393  , p=0.9998  , df=4
parameter F test:         F=1.5203  , p=0.1965  , df_denom=268, df_num=4
"""

# I also ran the diffLogEUR against other lags in the dataframe and they all had
# high p-values ranging from 0,2 - 1, so, not statistically meaningful.
# This is almost an inconclusive result. Although p-values for the first lag are 
# noticeably lower than the other lags and very close to 0.05, they don't trully
# reach 0.05 or surpass it, meaning I cannot reject the null hypothesis, which in
# turn means, "lag 1" doesn't have significant predictive power over "diffLogEUR".
# Considering the f-test get's to a maximum of 0.06, there's a chance that "lag 1"
# could help improve the predictive power, but that improvement could be insignificant
# I think the way here would be to test both models, with and without "lag 1" by
# running a VARIMA or ARIMAX model and comparing the statistics against a normal ARIMA.

# In[3.4]: As an experiment, running the EUR against USD:

df_wg_usd_1year = pd.read_csv("./datasets/wrangled/df_usd_1year.csv", float_precision="high", parse_dates=([0]))

df_wg_usd_1year['diffLogUSD'] = df_wg_usd_1year['logUSD'].diff()

df_eur_usd = pd.DataFrame({"eur": df_wg_eur_1year['diffLogEUR'], "usd": df_wg_usd_1year["diffLogUSD"]})

eur_usd_granger = grangercausalitytests(df_eur_usd[['eur', 'usd']].dropna(), maxlag=40)

"""
Granger Causality
number of lags (no zero) 1
ssr based F test:         F=1.1155  , p=0.2918  , df_denom=275, df_num=1
ssr based chi2 test:   chi2=1.1277  , p=0.2883  , df=1
likelihood ratio test: chi2=1.1254  , p=0.2888  , df=1
parameter F test:         F=1.1155  , p=0.2918  , df_denom=275, df_num=1

Granger Causality
number of lags (no zero) 2
ssr based F test:         F=0.5894  , p=0.5554  , df_denom=272, df_num=2
ssr based chi2 test:   chi2=1.2005  , p=0.5487  , df=2
likelihood ratio test: chi2=1.1979  , p=0.5494  , df=2
parameter F test:         F=0.5894  , p=0.5554  , df_denom=272, df_num=2

Granger Causality
number of lags (no zero) 3
ssr based F test:         F=0.4765  , p=0.6989  , df_denom=269, df_num=3
ssr based chi2 test:   chi2=1.4667  , p=0.6900  , df=3
likelihood ratio test: chi2=1.4628  , p=0.6909  , df=3
parameter F test:         F=0.4765  , p=0.6989  , df_denom=269, df_num=3

Granger Causality
number of lags (no zero) 4
ssr based F test:         F=0.9612  , p=0.4293  , df_denom=266, df_num=4
ssr based chi2 test:   chi2=3.9748  , p=0.4094  , df=4
likelihood ratio test: chi2=3.9463  , p=0.4133  , df=4
parameter F test:         F=0.9612  , p=0.4293  , df_denom=266, df_num=4
"""

"""
Granger Causality
number of lags (no zero) 29
ssr based F test:         F=1.3245  , p=0.1362  , df_denom=191, df_num=29
ssr based chi2 test:   chi2=50.2767 , p=0.0084  , df=29
likelihood ratio test: chi2=45.8109 , p=0.0245  , df=29
parameter F test:         F=1.3245  , p=0.1362  , df_denom=191, df_num=29

Granger Causality
number of lags (no zero) 30
ssr based F test:         F=1.2946  , p=0.1534  , df_denom=188, df_num=30
ssr based chi2 test:   chi2=51.4411 , p=0.0087  , df=30
likelihood ratio test: chi2=46.7619 , p=0.0262  , df=30
parameter F test:         F=1.2946  , p=0.1534  , df_denom=188, df_num=30

Granger Causality
number of lags (no zero) 31
ssr based F test:         F=1.2480  , p=0.1864  , df_denom=185, df_num=31
ssr based chi2 test:   chi2=51.8631 , p=0.0108  , df=31
likelihood ratio test: chi2=47.0945 , p=0.0321  , df=31
parameter F test:         F=1.2480  , p=0.1864  , df_denom=185, df_num=31

Granger Causality
number of lags (no zero) 32
ssr based F test:         F=1.2467  , p=0.1853  , df_denom=182, df_num=32
ssr based chi2 test:   chi2=54.1439 , p=0.0085  , df=32
likelihood ratio test: chi2=48.9554 , p=0.0280  , df=32
parameter F test:         F=1.2467  , p=0.1853  , df_denom=182, df_num=32

Granger Causality
number of lags (no zero) 33
ssr based F test:         F=1.2354  , p=0.1929  , df_denom=179, df_num=33
ssr based chi2 test:   chi2=56.0272 , p=0.0074  , df=33
likelihood ratio test: chi2=50.4757 , p=0.0264  , df=33
parameter F test:         F=1.2354  , p=0.1929  , df_denom=179, df_num=33

Granger Causality
number of lags (no zero) 34
ssr based F test:         F=1.1933  , p=0.2303  , df_denom=176, df_num=34
ssr based chi2 test:   chi2=56.4770 , p=0.0091  , df=34
likelihood ratio test: chi2=50.8217 , p=0.0318  , df=34
parameter F test:         F=1.1933  , p=0.2303  , df_denom=176, df_num=34
"""

"""
Granger Causality
number of lags (no zero) 68
ssr based F test:         F=1.1837  , p=0.2385  , df_denom=74, df_num=68
ssr based chi2 test:   chi2=229.5071, p=0.0000  , df=68
likelihood ratio test: chi2=155.3104, p=0.0000  , df=68
parameter F test:         F=1.1837  , p=0.2385  , df_denom=74, df_num=68

Granger Causality
number of lags (no zero) 69
ssr based F test:         F=1.1499  , p=0.2799  , df_denom=71, df_num=69
ssr based chi2 test:   chi2=234.6711, p=0.0000  , df=69
likelihood ratio test: chi2=157.5477, p=0.0000  , df=69
parameter F test:         F=1.1499  , p=0.2799  , df_denom=71, df_num=69
"""

# We can see here that some of the tests, specially the ones based on chi2 have
# very low p-values, but our F-tests never get below 0,13, which strongly indicates
# there's no addition of predictive power when adding the usd time series. Although
# we could make a case for the low p-values of the likelihood tests, specially
# at higher lags (30-60). For the purpose of this work it doesn't matter though because
# we're trying to predict values to a max of 2 weeks ahead, and the granger test
# shows us that the prediction power of the usd time series will work best in the
# long range and not in the short one, like we intend.
# The first lags in these case are very high for all tests, ranging from 0.08 on
# lag 14 for likelihood to 0.7 on lag 3 for the f-test

# In[3.5]: Testing VIF for EUR and USD:

eur_usd_constants = add_constant(df_eur_usd.dropna())

eur_usd_vif = pd.DataFrame()

eur_usd_vif['vif'] = [variance_inflation_factor(eur_usd_constants.values, i) for i in range(eur_usd_constants.shape[1])]

eur_usd_vif['variable'] = eur_usd_constants.columns

eur_usd_vif

"""
        vif variable
0  1.007058    const
1  2.830555      eur
2  2.830555      usd
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
Lag: 1, P-value: 0.032899608460114325
Lag: 2, P-value: 0.053909745727470926
Lag: 3, P-value: 0.07467439307919739
Lag: 4, P-value: 0.09717629125776463

Lag: 1, P-value: 0.746040205028232
Lag: 2, P-value: 0.7302350722479102
Lag: 3, P-value: 0.8959632729276952
Lag: 4, P-value: 0.8004691624497777

Lag: 1, P-value: 0.8780035341257023
Lag: 2, P-value: 0.8299119949223216
Lag: 3, P-value: 0.9434554515108172
Lag: 4, P-value: 0.8991973872113583

Lag: 1, P-value: 0.7927351036819501
Lag: 2, P-value: 0.9019776178229749
Lag: 3, P-value: 0.9644520501824317
Lag: 4, P-value: 0.9347486772913017

Lag: 1, P-value: 0.24768796659655679
Lag: 2, P-value: 0.5001622095397216
Lag: 3, P-value: 0.7884718315877377
Lag: 4, P-value: 0.45363713798092054
"""

# We can see that p-values for the lags across train/test sections is not consistent
# So we can confidently assume that there's no causality between these two time series
# USD doesn't explain EUR or the relantionship between the two time series is sensitive
# to specific time windows or suffer influence from external factors