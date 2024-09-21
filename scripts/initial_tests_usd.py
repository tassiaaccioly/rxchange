# -*- coding: utf-8 -*-

# In[0]: Importação dos pacotes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import kpss

# In[0.1]: Importação dos dataframes

df_wg_usd = pd.read_csv("./datasets/wrangled/df_usd.csv", float_precision="high", parse_dates=([1]))

df_wg_usd = df_wg_usd.loc[:, ~df_wg_usd.columns.str.contains('^Unnamed')]

df_wg_usd.info()

# In[0.2]: Calculate Statistics for dataset

var_usd = np.var(df_wg_usd['usd'])
# 0.006558319695589602


# In[1.0]: Defining Stationarity

# In[1.1]: Augmented Dickey-Fuller Test

usd_adf = adfuller(df_wg_usd["usd"], maxlag=13, autolag="AIC")
usd_adf

# p-value is > 0.05 (0.16), DO NOT REJECT the null hypothesis and suggests the presence of a unit root (non-stationary)
# adf stats > 10%, DO NOT REJECT the null hypothesis and suggests the presence of a unit root (non-stationary)

"""
 (-2.3340152630375437,
 0.1612318910518078,
 0,
 182,
 {'1%': -3.4668001583460613,
  '5%': -2.8775552336674317,
  '10%': -2.5753075498128246},
 -781.4992404633625)
"""

usd_adf_ols = adfuller(df_wg_usd['usd'], maxlag=None, autolag="AIC", store=True, regresults=True)[-1].resols.summary()
usd_adf_ols

"""
                            OLS Regression Results
==============================================================================
Dep. Variable:                      y   R-squared:                       0.029
Model:                            OLS   Adj. R-squared:                  0.024
Method:                 Least Squares   F-statistic:                     5.448
Date:                Fri, 20 Sep 2024   Prob (F-statistic):             0.0207
Time:                        20:15:52   Log-Likelihood:                 418.92
No. Observations:                 182   AIC:                            -833.8
Df Residuals:                     180   BIC:                            -827.4
Df Model:                           1
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1            -0.0519      0.022     -2.334      0.021      -0.096      -0.008
const          0.2901      0.124      2.338      0.020       0.045       0.535
==============================================================================
Omnibus:                       31.535   Durbin-Watson:                   1.968
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               87.116
Skew:                           0.690   Prob(JB):                     1.21e-19
Kurtosis:                       6.096   Cond. No.                         396.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""


# In[1.2]: Plotting ACF and PACF to determine correct number of lags for usd

plt.figure(figsize=(12,6))
plot_acf(df_wg_usd['usd'], lags=13)
plt.show()

plt.figure(figsize=(12,6))
plot_pacf(df_wg_usd['usd'], lags=13)
plt.show()

# In[1.3]: Running KPSS test to determine stationarity for usd

usd_kpss = kpss(df_wg_usd['usd'], regression="c", nlags="auto")
usd_kpss

# p-value > 0.05 (0.1) DO NOT REJECT the null hypothesis, suggesting data is stationary
# adf stat of 0.262 is < than 0.463 (5%), DO NOT REJECT the null hypothesis, suggesting data is stationary

"""
(0.26288427948301424,
 0.1,
 9,
 {'10%': 0.347, '5%': 0.463, '2.5%': 0.574, '1%': 0.739})
"""


# In[2.0]: Detrending the data to make series stationary for usd

# We need to detrend instead of differencing because adf suggests non-stationary and 
# kpss suggests stationary, which suggests the presence of a trend

# In[2.1]: Determining the trend:
    
# Creating the constant:
    
constant = sm.add_constant(np.arange(len(df_wg_usd)))

# Run OLS model to get intercet

fitted_trend_usd = sm.OLS(df_wg_usd['usd'], constant).fit()

fitted_trend_usd.summary()

"""
                            OLS Regression Results
==============================================================================
Dep. Variable:                    usd   R-squared:                       0.015
Model:                            OLS   Adj. R-squared:                  0.009
Method:                 Least Squares   F-statistic:                     2.695
Date:                Fri, 20 Sep 2024   Prob (F-statistic):              0.102
Time:                        18:31:40   Log-Likelihood:                 201.66
No. Observations:                 183   AIC:                            -399.3
Df Residuals:                     181   BIC:                            -392.9
Df Model:                           1
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          5.5968      0.012    470.264      0.000       5.573       5.620
x1            -0.0002      0.000     -1.642      0.102      -0.000    3.75e-05
==============================================================================
Omnibus:                       12.401   Durbin-Watson:                   0.093
Prob(Omnibus):                  0.002   Jarque-Bera (JB):                4.912
Skew:                          -0.037   Prob(JB):                       0.0858
Kurtosis:                       2.201   Cond. No.                         210.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

# In[2.2]: Remove the trend value from the data:

df_wg_usd['detrendUSD'] = df_wg_usd['usd'] - fitted_trend_usd.fittedvalues


# In[3.0]: Running ADF again with detrended value to see if series is now stationary for usd

usd_adf_det = adfuller(df_wg_usd["detrendUSD"], maxlag=13, autolag="AIC")
usd_adf_det

# p-value is > 0.05 (0.13), DO NOT REJECT the null hypothesis and suggests the presence of a unit root (non-stationary)
# adf stats > 10%, DO NOT REJECT the null hypothesis and suggests the presence of a unit root (non-stationary)

"""
(-2.4419863639013895,
 0.13023492155830607,
 0,
 182,
 {'1%': -3.4668001583460613,
  '5%': -2.8775552336674317,
  '10%': -2.5753075498128246},
 -782.3479509939586)
"""

usd_adf_det_ols = adfuller(df_wg_usd['detrendUSD'], maxlag=None, autolag="AIC", store=True, regresults=True)[-1].resols.summary()
usd_adf_det_ols

"""
                            OLS Regression Results
==============================================================================
Dep. Variable:                      y   R-squared:                       0.032
Model:                            OLS   Adj. R-squared:                  0.027
Method:                 Least Squares   F-statistic:                     5.963
Date:                Fri, 20 Sep 2024   Prob (F-statistic):             0.0156
Time:                        19:04:41   Log-Likelihood:                 419.17
No. Observations:                 182   AIC:                            -834.3
Df Residuals:                     180   BIC:                            -827.9
Df Model:                           1
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1            -0.0546      0.022     -2.442      0.016      -0.099      -0.010
const          0.0008      0.002      0.426      0.671      -0.003       0.004
==============================================================================
Omnibus:                       30.999   Durbin-Watson:                   1.968
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               84.828
Skew:                           0.680   Prob(JB):                     3.80e-19
Kurtosis:                       6.055   Cond. No.                         12.4
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

# In[3.1]: Running KPSS again for detrendUSD

usd_det_kpss = kpss(df_wg_usd['detrendUSD'], regression="c", nlags="auto")
usd_det_kpss

# p-value > 0.05 (0.1) suggests we should not reject the null hypothesis, thus suggesting the series is stationary
# adf stat of 0.21 is > 10%, we do not reject the null hypothesis at 95%, also suggesting a stationary dataset

"""
(0.21554541234506638,
 0.1,
 9,
 {'10%': 0.347, '5%': 0.463, '2.5%': 0.574, '1%': 0.739})
"""

# In[4.0]: Differencing the data

df_wg_usd['diffUSD'] = df_wg_usd['usd'].diff()

df_wg_usd

# In[5.0]: Running the ADF test again:
    
usd_adf_diff = adfuller(df_wg_usd["diffUSD"].dropna(), maxlag=13, autolag="AIC")
usd_adf_diff

# p-value is < 0.05, REJECTS the null hypothesis and suggests the absence of a unit root (Stationary)
# adf stats < 1%, REJECTS the null hypothesis and suggests the absence of a unit root (Stationary)

"""
(-13.550891814937843,
 2.410672989858822e-25,
 0,
 181,
 {'1%': -3.467004502498507,
  '5%': -2.8776444997243558,
  '10%': -2.575355189707274},
 -778.0804205067669)
"""

usd_adf_diff_ols = adfuller(df_wg_usd['diffUSD'].dropna(), maxlag=None, autolag="AIC", store=True, regresults=True)[-1].resols.summary()
usd_adf_diff_ols

"""
                            OLS Regression Results
==============================================================================
Dep. Variable:                      y   R-squared:                       0.506
Model:                            OLS   Adj. R-squared:                  0.504
Method:                 Least Squares   F-statistic:                     183.6
Date:                Fri, 20 Sep 2024   Prob (F-statistic):           3.01e-29
Time:                        20:30:26   Log-Likelihood:                 414.04
No. Observations:                 181   AIC:                            -824.1
Df Residuals:                     179   BIC:                            -817.7
Df Model:                           1
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1            -1.0094      0.074    -13.551      0.000      -1.156      -0.862
const          0.0004      0.002      0.239      0.812      -0.003       0.004
==============================================================================
Omnibus:                       27.384   Durbin-Watson:                   2.003
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               78.141
Skew:                           0.576   Prob(JB):                     1.08e-17
Kurtosis:                       6.006   Cond. No.                         40.6
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""