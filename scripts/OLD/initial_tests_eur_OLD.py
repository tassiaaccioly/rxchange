# In[0.1]: Importação dos pacotes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import kpss

# In[0.2]: Import dataframes

df_wg_eur = pd.read_csv("./datasets/wrangled/df_eur.csv", float_precision="high", parse_dates=([1]))

df_wg_eur = df_wg_eur.loc[:, ~df_wg_eur.columns.str.contains('^Unnamed')]

df_wg_eur.info()

# In[0.3]: Calculate Statistics for datasets

var_eur = np.var(df_wg_eur['eur'])
# 0.006964054485861109

# In[1.0]: Defining Stationarity

# In[1.1]: Augmented Dickey-Fuller Test

eur_adf = adfuller(df_wg_eur["eur"], maxlag=None, autolag="AIC")
eur_adf

# p-value is > 0.05 (0.08), DO NOT REJECT the null hypothesis and suggests the presence of a unit root (NON-STATIONARY)
# adf stats > 5% < 10%, DO NOT REJECT the null hypothesis and suggests the presence of a unit root (NON-STATIONARY)

"""
(-2.645994127964288,
 0.0838588367217662,
 0,
 179,
 {'1%': -3.4674201432469816,
  '5%': -2.877826051844538,
  '10%': -2.575452082332012},
 -723.2729790069569)
"""

eur_adf_ols = adfuller(df_wg_eur['eur'], maxlag=None, autolag="AIC", store=True, regresults=True)[-1].resols.summary()
eur_adf_ols

"""
                            OLS Regression Results
==============================================================================
Dep. Variable:                      y   R-squared:                       0.038
Model:                            OLS   Adj. R-squared:                  0.033
Method:                 Least Squares   F-statistic:                     7.001
Date:                Fri, 20 Sep 2024   Prob (F-statistic):            0.00888
Time:                        20:32:01   Log-Likelihood:                 391.66
No. Observations:                 179   AIC:                            -779.3
Df Residuals:                     177   BIC:                            -772.9
Df Model:                           1
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1            -0.0645      0.024     -2.646      0.009      -0.113      -0.016
const          0.3967      0.150      2.653      0.009       0.102       0.692
==============================================================================
Omnibus:                       39.197   Durbin-Watson:                   2.122
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              117.546
Skew:                           0.854   Prob(JB):                     2.99e-26
Kurtosis:                       6.583   Cond. No.                         462.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

# In[1.2]: Plotting ACF and PACF to determine correct number of lags for eur

plt.figure(figsize=(12,6))
plot_acf(df_wg_eur['eur'], lags=13)
plt.show()

plt.figure(figsize=(12,6))
plot_pacf(df_wg_eur['eur'], lags=13)
plt.show()

# Correct number of lags from both is 0;

# In[1.3]: Running KPSS test to determine stationarity for eur

eur_kpss = kpss(df_wg_eur['eur'], regression="c", nlags="auto")
eur_kpss

# p-value < 0.05 (0.033) REJECTS the null hypothesis, suggesting data is NON-STATIONARY
# kpss stat of 0.534 is > than 0.463 (5%), REJECTS the null hypothesis, suggesting data is NON-STATIONARY

"""
(0.5346460747318406,
 0.03386349668201789,
 9,
 {'10%': 0.347, '5%': 0.463, '2.5%': 0.574, '1%': 0.739})
"""

# In[2.0]: Applying differencing to make series stationary for eur

df_wg_eur['diffEUR'] = df_wg_eur['eur'].diff()

df_wg_eur

# In[2.1]: Running ADF again to see if series is stationary for eur

eur_adf_diff = adfuller(df_wg_eur["diffEUR"].dropna(), maxlag=None, autolag="AIC")
eur_adf_diff

# p-value < 0.05, REJECTS the null hypothesis, suggesting there's no unit root  so data is STATIONARY
# adf stat < 1% , REJECTS the null hypothesis, suggesting there's no unit root so data is STATIONARY

"""
(-14.556984704321536,
 4.863192311189012e-27,
 0,
 178,
 {'1%': -3.467631519151906,
  '5%': -2.8779183721695567,
  '10%': -2.575501353364474},
 -713.9807033122561)
"""

eur_adf_diff_ols = adfuller(df_wg_eur['diffEUR'].dropna(), maxlag=None, autolag="AIC", store=True, regresults=True)[-1].resols.summary()
eur_adf_diff_ols

"""
                            OLS Regression Results
==============================================================================
Dep. Variable:                      y   R-squared:                       0.546
Model:                            OLS   Adj. R-squared:                  0.544
Method:                 Least Squares   F-statistic:                     211.9
Date:                Fri, 20 Sep 2024   Prob (F-statistic):           5.07e-32
Time:                        20:35:22   Log-Likelihood:                 386.59
No. Observations:                 178   AIC:                            -769.2
Df Residuals:                     176   BIC:                            -762.8
Df Model:                           1
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1            -1.0907      0.075    -14.557      0.000      -1.239      -0.943
const          0.0010      0.002      0.497      0.620      -0.003       0.005
==============================================================================
Omnibus:                       34.283   Durbin-Watson:                   2.011
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              101.954
Skew:                           0.742   Prob(JB):                     7.26e-23
Kurtosis:                       6.397   Cond. No.                         36.0
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

# In[2.2]: Running KPSS again for diffEUR

eur_diff_kpss = kpss(df_wg_eur['diffEUR'].dropna(), regression="c", nlags="auto")
eur_diff_kpss

# p-value > 0.05 (0.1) DO NOT REJECT the null hypothesis, suggesting the data is STATIONARY
# kpss stat < 10% (0.08), DO NOT REJECT the null hypothesis, suggesting the data is STATIONARY

"""
(0.08982787925254508,
 0.1,
 4,
 {'10%': 0.347, '5%': 0.463, '2.5%': 0.574, '1%': 0.739})
"""

# In[3]: Plotting ACF and PACF to determine ppp and qqq for ARIMA

plt.figure(figsize=(12,6))
plot_acf(df_wg_eur['diffEUR'], lags=13)
plt.show()

plt.figure(figsize=(12,6))
plot_pacf(df_wg_eur['diffEUR'], lags=13)
plt.show()

