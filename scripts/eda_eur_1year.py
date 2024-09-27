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
from scipy.stats import boxcox, norm
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime


# In[0.2]: Import dataframes

df_wg_eur_1year = pd.read_csv("./datasets/wrangled/df_eur_1year.csv", float_precision="high", parse_dates=([1]))

df_wg_eur_1year.info()

df_wg_eur_5year = pd.read_csv("./datasets/wrangled/df_eur_5year.csv", float_precision="high", parse_dates=([1]))

df_wg_eur_5year

dt_format = "%d/%m/%Y"

df_wg_eur_1year["sqrt"] = np.sqrt(df_wg_eur_1year["eur"])

df_wg_eur_1year = df_wg_eur_1year.drop(df_wg_eur_1year[df_wg_eur_1year["weekday"] == "Sunday"].index)

# In[0.3]: Plot the 5 year dataset

plt.figure(figsize=(15, 10), dpi=300)
sns.lineplot(x=df_wg_eur_5year["dateTime"], y=df_wg_eur_5year["eur"], color="limegreen", label="Câmbio EUR")
plt.axhline(y=np.mean(df_wg_eur_5year["eur"]), color="black", linestyle="--", label="Média", linewidth=2) # mean for euro
plt.axhline(y=np.max(df_wg_eur_5year["eur"]), color="magenta", label="Máxima", linewidth=2) # máxima for euro
plt.axhline(y=np.min(df_wg_eur_5year["eur"]), color="magenta", linestyle="--", label="Mínima", linewidth=2) # máxima for euro
plt.title(f'Cotação do Euro - Série histórica ({df_wg_eur_5year["dateTime"][0]} - {df_wg_eur_5year["dateTime"][1740]})', fontsize="18")
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

# 0.020854549842151036

varlog_eur_1year = np.var(df_wg_eur_1year['log'])
varlog_eur_1year

# 0.0006837815880977775

optimal_lags_eur_1year = 12*(len(df_wg_eur_1year['eur'])/100)**(1/4)
optimal_lags_eur_1year

# 14.935991454923482

# So optimal lags for this dataset = 14 or 15

# In[0.5]: Plot the original dataset for visual assessment

#plt.rcParams.update({'figure.figsize':(15,10), 'figure.dpi':120})

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

# In[0.6]: Plot the log dataset for visual assessment

plt.figure(figsize=(15, 10), dpi=300)
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

# Doing a box-cox transformation in the logarithmic data:

# this lamba value comes from R. guerrero method gave lambda -0.9999
# and method loglik gave lamba -1 (chose -1 because of easier interpretation)
# Why R was used: https://github.com/scipy/scipy/issues/6873

eur_boxcox_lambda = -1

eur_1year_boxcox = boxcox(df_wg_eur_1year["log"], eur_boxcox_lambda)

df_wg_eur_1year["boxcox"] = pd.DataFrame(eur_1year_boxcox)

# Plot box-cox:

fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(11,9), dpi=300)
df_wg_eur_1year["log"].plot(ax=axs[0], label="Log")
axs[0].set_title("Série Original")
df_wg_eur_1year["boxcox"].plot(ax=axs[1], label="Box-cox", color="orange")
axs[1].set_title("Box-Cox")
plt.suptitle("Série Original x Série Boxcox")
plt.tight_layout()
plt.show()

# Plot the histogram

mu, std = norm.fit(df_wg_eur_1year["boxcox"])

plt.style.use("grayscale")
plt.figure(figsize=(15, 10), dpi=300)
sns.histplot(x=df_wg_eur_1year["boxcox"], alpha=0.4,
             edgecolor=None, kde=True, line_kws={
                 "linewidth": 3, "linestyle": "dashed", "color": "m",
                 "label": "KDE"}, 
             )
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, len(df_wg_eur_1year["boxcox"]))
p = norm.pdf(x, mu, std)
plt.plot(x, p, linewidth=3, label="Distribuição Normal")
plt.title("Transformação de Box-Cox nos dados com Log", fontsize="18")
plt.legend(fontsize=18, loc="upper right")
plt.show()

# Plot scatter of the Box-cox values
plt.figure(figsize=(15, 10), dpi=300)
sns.scatterplot(df_wg_eur_1year["boxcox"], color="green")
plt.title("Transformação de Box-Cox nos dados com Log", fontsize="18")
plt.xlabel("")
plt.ylabel("")
plt.legend(fontsize=18, loc="upper right")
plt.show()

# Plotting the boxplot to see outliers in the boxcox dataset

plt.figure(figsize=(15,10), dpi=300)
sns.boxplot(x=df_wg_eur_1year["eur"], palette="viridis")
plt.title("Boxplot do Dataset Eur", fontsize="18")
plt.legend(fontsize="16")
plt.show()

plt.figure(figsize=(15,10), dpi=300)
sns.boxplot(x=df_wg_eur_1year["log"], palette="viridis")
plt.title("Boxplot do Dataset com Log", fontsize="18")
plt.legend(fontsize="16")
plt.show()

plt.figure(figsize=(15,10), dpi=300)
sns.boxplot(x=df_wg_eur_1year["boxcox"], palette="viridis")
plt.title("Boxplot do Dataset com BoxCox", fontsize="18")
plt.legend(fontsize="16")
plt.show()

plt.figure(figsize=(15,10), dpi=300)
sns.boxplot(x=np.sqrt(df_wg_eur_1year["boxcox"]), palette="viridis")
plt.title("Boxplot do Dataset com Raiz de BoxCox", fontsize="18")
plt.legend(fontsize="16")
plt.show()

plt.figure(figsize=(15,10), dpi=300)
sns.boxplot(x=df_wg_eur_1year["sqrt"], palette="viridis")
plt.title("Boxplot do Dataset com Raiz", fontsize="18")
plt.legend(fontsize="16")
plt.show()


# In[1.1]: Augmented Dickey-Fuller Test with eur and log

eur_1year_adf_ols = adfuller(df_wg_eur_1year['eur'], maxlag=None, autolag="AIC", store=True, regresults=True)[-1].resols.summary()
eur_1year_adf_ols

# Normal values results:
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.070
Model:                            OLS   Adj. R-squared:                  0.041
Method:                 Least Squares   F-statistic:                     2.434
Date:                Thu, 26 Sep 2024   Prob (F-statistic):             0.0201
Time:                        18:37:26   Log-Likelihood:                 498.90
No. Observations:                 233   AIC:                            -981.8
Df Residuals:                     225   BIC:                            -954.2
Df Model:                           7                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1             0.0209      0.015      1.411      0.160      -0.008       0.050
x2            -0.1364      0.069     -1.990      0.048      -0.271      -0.001
x3            -0.1184      0.069     -1.725      0.086      -0.254       0.017
x4            -0.2134      0.069     -3.078      0.002      -0.350      -0.077
x5            -0.0031      0.069     -0.044      0.965      -0.139       0.133
x6            -0.0781      0.069     -1.136      0.257      -0.214       0.057
x7            -0.1544      0.068     -2.264      0.025      -0.289      -0.020
const         -0.1092      0.080     -1.362      0.175      -0.267       0.049
==============================================================================
Omnibus:                       13.399   Durbin-Watson:                   1.990
Prob(Omnibus):                  0.001   Jarque-Bera (JB):               25.011
Skew:                           0.278   Prob(JB):                     3.71e-06
Kurtosis:                       4.506   Cond. No.                         298.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

# Doing the same for the log database

log_1year_adf_ols = adfuller(df_wg_eur_1year['log'], maxlag=None, autolag="AIC", store=True, regresults=True)[-1].resols.summary()
log_1year_adf_ols

# Results for log values:
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.066
Model:                            OLS   Adj. R-squared:                  0.037
Method:                 Least Squares   F-statistic:                     2.286
Date:                Thu, 26 Sep 2024   Prob (F-statistic):             0.0287
Time:                        18:38:27   Log-Likelihood:                 894.50
No. Observations:                 233   AIC:                            -1773.
Df Residuals:                     225   BIC:                            -1745.
Df Model:                           7                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1             0.0187      0.015      1.254      0.211      -0.011       0.048
x2            -0.1325      0.068     -1.937      0.054      -0.267       0.002
x3            -0.1204      0.068     -1.758      0.080      -0.255       0.015
x4            -0.2005      0.069     -2.899      0.004      -0.337      -0.064
x5            -0.0042      0.069     -0.061      0.951      -0.140       0.131
x6            -0.0719      0.069     -1.049      0.295      -0.207       0.063
x7            -0.1525      0.068     -2.244      0.026      -0.286      -0.019
const         -0.0309      0.025     -1.226      0.221      -0.080       0.019
==============================================================================
Omnibus:                       13.277   Durbin-Watson:                   1.992
Prob(Omnibus):                  0.001   Jarque-Bera (JB):               24.635
Skew:                           0.277   Prob(JB):                     4.47e-06
Kurtosis:                       4.494   Cond. No.                         511.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

# Doing the same for the box-cox transformed database

log_1year_adf_ols = adfuller(df_wg_eur_1year['boxcox'], maxlag=None, autolag="AIC", store=True, regresults=True)[-1].resols.summary()
log_1year_adf_ols

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.047
Model:                            OLS   Adj. R-squared:                  0.018
Method:                 Least Squares   F-statistic:                     1.613
Date:                Fri, 27 Sep 2024   Prob (F-statistic):              0.121
Time:                        16:27:24   Log-Likelihood:                 1350.2
No. Observations:                 272   AIC:                            -2682.
Df Residuals:                     263   BIC:                            -2650.
Df Model:                           8                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1             0.0135      0.013      1.040      0.299      -0.012       0.039
x2            -0.1088      0.063     -1.726      0.085      -0.233       0.015
x3            -0.1432      0.063     -2.272      0.024      -0.267      -0.019
x4            -0.1016      0.064     -1.594      0.112      -0.227       0.024
x5            -0.0862      0.064     -1.353      0.177      -0.212       0.039
x6            -0.0226      0.063     -0.357      0.722      -0.148       0.102
x7            -0.0546      0.063     -0.866      0.387      -0.179       0.070
x8            -0.1359      0.063     -2.170      0.031      -0.259      -0.013
const         -0.0053      0.005     -1.002      0.317      -0.016       0.005
==============================================================================
Omnibus:                       23.751   Durbin-Watson:                   1.991
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               60.663
Skew:                           0.365   Prob(JB):                     6.72e-14
Kurtosis:                       5.196   Cond. No.                         863.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

# We can notice that in the third model, both AIC and BIC are lower than the first one
# and also that the loglike is higher on the third model than the others, proving
# that the boxcox version of the data is a better fit for the model so far.

# Getting values from the second model:

log_1year_adf = adfuller(df_wg_eur_1year["boxcox"], maxlag=None, autolag="AIC")
log_1year_adf

# p-value is > 0.05 (0.99), DO NOT REJECT the null hypothesis and suggests the presence of a unit root (NON-STATIONARY)
# adf stats > 10% (1.03), DO NOT REJECT the null hypothesis and suggests the presence of a unit root (NON-STATIONARY)
# lags = 7 good amount of lags

"""
(1.039857714950727,
 0.9946691892139733,
 7,
 272,
 {'1%': -3.4546223782586534,
  '5%': -2.8722253212300277,
  '10%': -2.5724638500216264},
 -2605.836821321932)
"""

# In[1.2]: Running KPSS test to determine stationarity for eur

log_1year_kpss = kpss(df_wg_eur_1year['boxcox'], regression="c", nlags="auto")
log_1year_kpss

# p-value < 0.05 (0.01) REJECTS the null hypothesis, suggesting data is NON-STATIONARY
# kpss stat of 2.02 is > than 1% (0.739), REJECTS the null hypothesis, suggesting data is NON-STATIONARY
# lags = 10 good amount of lags

"""
(2.024630025053029,
 0.01,
 10,
 {'10%': 0.347, '5%': 0.463, '2.5%': 0.574, '1%': 0.739})
"""

# In[1.3]: Plotting ACF and PACF to examine autocorrelations in data:

plt.style.use("seaborn-v0_8-colorblind")
fig, (ax1, ax2) = plt.subplots(2, figsize=(11, 7), dpi=300, sharex=True)
plot_acf(df_wg_eur_1year["boxcox"], ax=ax1)
plot_pacf(df_wg_eur_1year["boxcox"], ax=ax2)
plt.show()

# PACF plot shows an AR(2) order for the dataset showing a high statistical significance spike 
# at the first lag in PACF. ACF shows slow decay towards 0 -> exponentially decaying or sinusoidal

# In[1.4]: Defining the order of differencing we need:

# Original Series
sns.reset_defaults()
fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(11,8), dpi=300)
ax1.plot(df_wg_eur_1year["boxcox"]);
ax1.set_title('Série Original log(EUR)');

# 1st Differencing
ax2.plot(df_wg_eur_1year["boxcox"].diff());
ax2.set_title('1ª Ordem de Diferenciação');

# 2nd Differencing
ax3.plot(df_wg_eur_1year["boxcox"].diff().diff());
ax3.set_title('2ª Ordem de Diferenciação') 
plt.show()


# Plotting the ACF for each order

fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(11, 10), dpi=300)
plot_acf(df_wg_eur_1year["boxcox"], ax=ax1)
plot_acf(df_wg_eur_1year["boxcox"].diff().dropna(), ax=ax2)
plot_acf(df_wg_eur_1year["boxcox"].diff().diff().dropna(), ax=ax3)
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
    
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(11, 10), dpi=300)
plot_pacf(df_wg_eur_1year["log"], ax=ax1)
plot_pacf(df_wg_eur_1year["diff"].diff().dropna(), ax=ax2, color="green")
plot_pacf(df_wg_eur_1year["diff"].diff().dropna(), ax=ax3)
plt.show()

# Plotting ACF and PACF together:

plt.figure(figsize=(11,10), dpi=300)
fig, (ax1, ax2) = plt.subplots(2)
plot_acf(df_wg_eur_1year["diff"].dropna(), ax=ax1)
plot_pacf(df_wg_eur_1year["diff"].dropna(), ax=ax2)

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
sns.lineplot(x=df_wg_eur_1year["dateTime"], y=df_wg_eur_1year["diff"], color="limegreen", label="Log(EUR) - Diff")
plt.axhline(y=np.mean(df_wg_eur_1year["diff"]), color="black", linestyle="--", label="Média") # mean for eur
plt.title(f'Log do Euro Diferenciado - Série histórica ({df_wg_eur_1year["dateTime"][0].strftime(dt_format)} - {df_wg_eur_1year["dateTime"][239].strftime(dt_format)})', fontsize="18")
plt.yticks(np.arange(round(df_wg_eur_1year["diff"].min(), 3), round(df_wg_eur_1year["diff"].max() + 0.002, 3), 0.002), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b, %Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=29))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Log(Câmbio EUR ↔ BRL) Diferenciado ", fontsize="18")
plt.legend(fontsize=18, loc="upper center")
plt.show()

# In[2.0]: Defining Stationarity again

optimal_lags_eur_1year_diff = 12*(len(df_wg_eur_1year['diff'].dropna())/100)**(1/4)
optimal_lags_eur_1year_diff

# 14.920408761353036

# So optimal lags for this dataset = 14 or 15

# In[2.1]: Augmented Dickey-Fuller Test with diff

log_1year_diff_adf = adfuller(df_wg_eur_1year["diff"].dropna(), maxlag=None, autolag="AIC")
log_1year_diff_adf

# p-value is <<< 0.05 (5.988728689379452e-20), REJECTS the null hypothesis and suggests the absence of a unit root (STATIONARY)
# adf stats <<< 1% (-11), REJECTS the null hypothesis and suggests the absence of a unit root (STATIONARY)
# lags = 2 good amount of lags

"""
(-11.02098257712512,
 5.988728689379452e-20,
 2,
 236,
 {'1%': -3.4583663275730476,
  '5%': -2.8738660999177132,
  '10%': -2.5733390785693766},
 -1711.895379163122)
"""

log_1year_diff_adf_ols = adfuller(df_wg_eur_1year['diff'].dropna(), maxlag=None, autolag="AIC", store=True, regresults=True)[-1].resols.summary()
log_1year_diff_adf_ols

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.555
Model:                            OLS   Adj. R-squared:                  0.549
Method:                 Least Squares   F-statistic:                     96.42
Date:                Thu, 26 Sep 2024   Prob (F-statistic):           1.50e-40
Time:                        19:12:13   Log-Likelihood:                 903.52
No. Observations:                 236   AIC:                            -1799.
Df Residuals:                     232   BIC:                            -1785.
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1            -1.3701      0.124    -11.021      0.000      -1.615      -1.125
x2             0.2610      0.098      2.668      0.008       0.068       0.454
x3             0.1591      0.066      2.416      0.016       0.029       0.289
const          0.0006      0.000      1.831      0.068   -4.86e-05       0.001
==============================================================================
Omnibus:                       13.164   Durbin-Watson:                   1.974
Prob(Omnibus):                  0.001   Jarque-Bera (JB):               23.891
Skew:                           0.280   Prob(JB):                     6.49e-06
Kurtosis:                       4.455   Cond. No.                         463.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

# AIC, BIC and Loglike prove this model is even better than the second one, so
# I SHOULD USE D = 1 IN THE ARIMA MODEL

# In[2.2]: Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test post differencing

log_1year_diff_kpss = kpss(df_wg_eur_1year['diff'].dropna(), regression="c", nlags="auto")
log_1year_diff_kpss

# p-value > 0.05 (0.1) DO NOT REJECT the null hypothesis, suggesting data is STATIONARY
# kpss stat of 0.32 is < 10% (0.347), DO NOT REJECT the null hypothesis, suggesting data is STATIONARY
# lags = 15 ok amount of lags for this point in time

"""
(0.3264805776862086,
 0.1,
 15,
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

plt.figure(figsize=(11, 8), dpi=300)
fig, (ax1, ax2) = plt.subplots(2)
plot_acf(df_wg_eur_1year["diff"].dropna(), ax=ax1)
plot_pacf(df_wg_eur_1year["diff"].dropna(), ax=ax2)

# Both tests seem to show normal stationary plots with no significant lags after zero.
# Also, no significant negative first lag;

# In[3.0]: Running Granger Causality tests

# In[3.1]: Create the the lagged dataframe:

eur_1year_lagged = pd.DataFrame({"log": df_wg_eur_1year["log"]})

eur_1year_lagged['lag 1'] = eur_1year_lagged["log"].shift(1)
eur_1year_lagged['lag 2'] = eur_1year_lagged["log"].shift(2)
eur_1year_lagged['lag 3'] = eur_1year_lagged["log"].shift(3)
eur_1year_lagged['lag 4'] = eur_1year_lagged["log"].shift(4)
eur_1year_lagged['lag 5'] = eur_1year_lagged["log"].shift(5)
eur_1year_lagged['lag 6'] = eur_1year_lagged["log"].shift(6)

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
0  5077.036898    const
1    24.506497      log
2    42.658034    lag 1
3    41.216375    lag 2
4    40.114452    lag 3
5    39.847614    lag 4
6    39.000550    lag 5
7    21.746609    lag 6
"""

# We can see very high VIFs between the lags; Which means there's multicolinearity

# In[3.3]: Running the actual Causality test on the lagged data

eur_granger_lag3 = grangercausalitytests(eur_1year_lagged[["log", "lag 3"]].dropna(), maxlag=4)

"""
Granger Causality
number of lags (no zero) 1
ssr based F test:         F=9.3256  , p=0.0025  , df_denom=233, df_num=1
ssr based chi2 test:   chi2=9.4457  , p=0.0021  , df=1
likelihood ratio test: chi2=9.2615  , p=0.0023  , df=1
parameter F test:         F=9.3256  , p=0.0025  , df_denom=233, df_num=1

Granger Causality
number of lags (no zero) 2
ssr based F test:         F=3.8073  , p=0.0236  , df_denom=230, df_num=2
ssr based chi2 test:   chi2=7.7801  , p=0.0204  , df=2
likelihood ratio test: chi2=7.6541  , p=0.0218  , df=2
parameter F test:         F=3.8073  , p=0.0236  , df_denom=230, df_num=2
"""

eur_granger_lag4 = grangercausalitytests(eur_1year_lagged[["log", "lag 4"]].dropna(), maxlag=4)

"""
Granger Causality
number of lags (no zero) 1
ssr based F test:         F=4.8395  , p=0.0288  , df_denom=232, df_num=1
ssr based chi2 test:   chi2=4.9021  , p=0.0268  , df=1
likelihood ratio test: chi2=4.8517  , p=0.0276  , df=1
parameter F test:         F=4.8395  , p=0.0288  , df_denom=232, df_num=1
"""

eur_granger_lag5 = grangercausalitytests(eur_1year_lagged[["log", "lag 5"]].dropna(), maxlag=4)

"""
Granger Causality
number of lags (no zero) 1
ssr based F test:         F=5.2387  , p=0.0230  , df_denom=231, df_num=1
ssr based chi2 test:   chi2=5.3067  , p=0.0212  , df=1
likelihood ratio test: chi2=5.2474  , p=0.0220  , df=1
parameter F test:         F=5.2387  , p=0.0230  , df_denom=231, df_num=1

Granger Causality
number of lags (no zero) 2
ssr based F test:         F=4.3410  , p=0.0141  , df_denom=228, df_num=2
ssr based chi2 test:   chi2=8.8724  , p=0.0118  , df=2
likelihood ratio test: chi2=8.7076  , p=0.0129  , df=2
parameter F test:         F=4.3410  , p=0.0141  , df_denom=228, df_num=2
"""

eur_granger_lag6 = grangercausalitytests(eur_1year_lagged[["log", "lag 6"]].dropna(), maxlag=4)

"""
Granger Causality
number of lags (no zero) 1
ssr based F test:         F=10.1024 , p=0.0017  , df_denom=230, df_num=1
ssr based chi2 test:   chi2=10.2342 , p=0.0014  , df=1
likelihood ratio test: chi2=10.0158 , p=0.0016  , df=1
parameter F test:         F=10.1024 , p=0.0017  , df_denom=230, df_num=1

Granger Causality
number of lags (no zero) 2
ssr based F test:         F=4.2149  , p=0.0159  , df_denom=227, df_num=2
ssr based chi2 test:   chi2=8.6154  , p=0.0135  , df=2
likelihood ratio test: chi2=8.4593  , p=0.0146  , df=2
parameter F test:         F=4.2149  , p=0.0159  , df_denom=227, df_num=2
"""

# Choosing "lag 6" as it has the best p-values.

# I've run other amount of lags for all of these, but mostly have p-values close
# to 1.

# In[3.4]: As an experiment, running the EUR against USD:

usd_1year_granger = pd.read_csv("./datasets/wrangled/df_usd_1year.csv", float_precision="high", parse_dates=([0]))

df_eur_usd = pd.DataFrame({"eur": df_wg_eur_1year['log'], "usd": usd_1year_granger["log"]})

eur_usd_granger = grangercausalitytests(df_eur_usd[['eur', 'usd']].dropna(), maxlag=40)

# There are no significant lags for eur and usd.

# Comment from previous tests
# We can see here that some of the tests, specially the ones based on chi2 have
# very low p-values, but our F-tests never get below 0,12, which strongly indicates
# there's no addition of predictive power when adding the usd time series. Although
# we could make a case for the low p-values of the likelihood tests, specially
# at higher lags (28-32). For the purpose of this work it doesn't make sense because
# we're trying to predict values to a max of 2 weeks ahead, and the granger test
# shows us that the prediction power of the usd time series would work best in the
# long term and not in the short term, like we intend.

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


# Similar results for this test, no significant lags
# We can see that p-values for the lags across train/test sections is not consistent
# mostly losing any potential influencial power in later splits whem compared to the
# first one. So we can confidently assume that there's no causality between these two
# time series USD doesn't explain EUR or the relantionship between the two time series
# is sensitive to specific time windows or suffer influence from external factors

# In[4.0]: Save final dataset for testing ARIMA

eur_arima_1year = pd.DataFrame({
    "date": df_wg_eur_1year["dateTime"],
    "eur": df_wg_eur_1year["eur"],
    "log": df_wg_eur_1year["log"],
    "diff": df_wg_eur_1year["diff"],
    "lag6": eur_1year_lagged["lag 6"]
    })

# save to csv
eur_arima_1year.to_csv("./datasets/arima_ready/eur_arima_1year.csv", index=False)
