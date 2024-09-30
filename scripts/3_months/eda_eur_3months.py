# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # #
# Exploratory Data Analysis = EURO - 365 days #
# # # # # # # # # # # # # # # # # # # # # # # #

# In[0.1]: Importação dos pacotes

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import boxcox, norm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.model_selection import TimeSeriesSplit


# In[0.2]: Import dataframes

df_eur_3months = pd.read_csv("./datasets/wrangled/df_eur_3months.csv", float_precision="high", parse_dates=([1]))

df_eur_3months.info()

df_wg_eur_3months = df_eur_3months

dt_format = "%d/%m/%Y"

df_wg_eur_3months = df_wg_eur_3months.drop(df_wg_eur_3months[df_wg_eur_3months["weekday"] == "Sunday"].index)

df_wg_eur_3months = df_wg_eur_3months.drop(df_wg_eur_3months[df_wg_eur_3months["weekday"] == "Saturday"].index)[1:]

df_wg_eur_3months = df_wg_eur_3months.drop(["weekday", "date"], axis=1)

df_wg_eur_3months = pd.DataFrame(df_wg_eur_3months.drop("dateTime", axis=1).values, index=pd.date_range(start="2024-06-24", periods=65 ,freq="B"), columns=["eur", "log", "diff"])


# In[0.3]: Separate data into train and test

# remove last 15 days of data for test later
eur_train = df_wg_eur_3months[:-20]

eur_test = df_wg_eur_3months[-20:]

# In[0.4]: Plot original datasets

# 3 months dataset
sns.set_style("whitegrid",  {"grid.linestyle": ":"})
plt.figure(figsize=(15, 10), dpi=300)
sns.lineplot(eur_train["eur"], color="limegreen", label="Câmbio EUR")
plt.axhline(y=np.mean(eur_train["eur"]), color="black", linestyle="--", label="Média", linewidth=2) # mean for euro
plt.axhline(y=np.max(eur_train["eur"]), color="magenta", label="Máxima", linewidth=2) # máxima for euro
plt.axhline(y=np.min(eur_train["eur"]), color="magenta", linestyle="--", label="Mínima", linewidth=2) # máxima for euro
plt.title(f'Cotação do Euro - Série histórica ({df_eur_3months["dateTime"][0].strftime(dt_format)} - {df_eur_3months["dateTime"][45].strftime(dt_format)})', fontsize="18")
plt.yticks(np.arange(round(eur_train["eur"].min(), 1), round(eur_train["eur"].max() + 0.1, 1), 0.1), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d, %b'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Câmbio EUR ↔ BRL", fontsize="18")
plt.legend(fontsize=18, loc="upper left", bbox_to_anchor=(0.02, 0, 0, 0.93))
plt.show()

# Boxplot for 3months:

plt.figure(figsize=(15,10), dpi=300)
sns.boxplot(x=eur_train["eur"], palette="viridis")
plt.title("Boxplot do Dataset - 1 ano (365 dias)", fontsize="18")
plt.legend(fontsize="16")
plt.show()


plt.figure(figsize=(15,10), dpi=300)
sns.boxplot(x=eur_train["diff"], palette="viridis")
plt.title("Boxplot do Dataset - 1 ano (365 dias)", fontsize="18")
plt.legend(fontsize="16")
plt.show()

# In[0.5]: Calculate Statistics for datasets

var_eur_3months = np.var(eur_train['eur'])
var_eur_3months

# 0.01334847399733333

varlog_eur_3months = np.var(eur_train['log'])
varlog_eur_3months

# 0.0003679907396973193

optimal_lags_eur_3months = 12*(len(eur_train['eur'])/100)**(1/4)
optimal_lags_eur_3months

# 9.82843510575264

# So optimal lags for this dataset = 9 or 10

# In[0.6]: Plot train and test datasets:

plt.figure(figsize=(15, 10), dpi=300)
sns.lineplot(eur_train["log"], color="limegreen", label="train log(Câmbio EUR)")
sns.lineplot(eur_test["log"], color="magenta", label="test log(Câmbio EUR)")
plt.axhline(y=np.mean(eur_train["log"]), color="black", linestyle="--", label="Média") # mean for euro
plt.axhline(y=np.max(eur_train["log"]), color="magenta", label="Máxima") # máxima for euro
plt.axhline(y=np.min(eur_train["log"]), color="magenta", linestyle="--", label="Mínima") # máxima for euro
plt.title(f'Cotação do Euro - Série histórica ({eur_train.index[0].strftime(dt_format)} - {eur_test.index[-1].strftime(dt_format)})', fontsize="18")
plt.yticks(np.arange(round(df_wg_eur_3months["log"].min(), 2), round(df_wg_eur_3months["log"].max() + 0.01, 2), 0.01), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d, %b'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Câmbio log(EUR ↔ BRL)", fontsize="18")
plt.legend(fontsize=18, loc="lower right", bbox_to_anchor=(0.99, 0.05, 0, 0))
plt.show()

# In[0.7]: Doing the seasonal decompose

eur_decom_multi = seasonal_decompose(eur_train["eur"], model="multiplicative")
eur_decom_multi.plot()

# Plotting season and residuals
plt.figure(figsize=(15,6))
eur_decom_multi.seasonal.plot(legend=True)
eur_decom_multi.resid.plot(legend=True)
plt.xlabel('Períodos',size=15)
plt.title('Sazonalidade e resíduos',size=15);

# plotting trend
plt.figure(figsize=(15,6))
eur_decom_multi.trend.plot(legend=True, color='r')
plt.xlabel('Períodos',size=15)
plt.title('Série de tendência',size=15);

eur_detrended = (eur_decom_multi.seasonal*eur_decom_multi.resid)
plt.figure(figsize=(15,6))
eur_detrended.plot()
plt.xlabel('Períodos',size=15)
plt.title('Gráfico da série detendencionalizada de forma multiplicativa',size=15);

eur_train['trend']=eur_decom_multi.trend
eur_train['detrend']=eur_train['eur']/eur_train['trend']
#gráfico das séries original e dessazonalizada
eur_train[['eur','detrend']].plot(figsize=(15,6))
plt.xlabel('Períodos',size=15)
plt.title('Série de consumo de cerveja com e sem sazonalidade',size=15);

# In[1.0]: Defining Stationarity

# In[1.1]: Augmented Dickey-Fuller Test with eur and log

eur_3months_adf_ols = adfuller(eur_train['eur'], maxlag=None, autolag="AIC", store=True, regresults=True)[-1].resols.summary()
eur_3months_adf_ols

# Normal values results:
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.183
Model:                            OLS   Adj. R-squared:                  0.119
Method:                 Least Squares   F-statistic:                     2.845
Date:                Sun, 29 Sep 2024   Prob (F-statistic):             0.0504
Time:                        19:10:07   Log-Likelihood:                 68.052
No. Observations:                  42   AIC:                            -128.1
Df Residuals:                      38   BIC:                            -121.2
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1            -0.1797      0.078     -2.315      0.026      -0.337      -0.023
x2             0.2506      0.155      1.615      0.115      -0.064       0.565
x3             0.1996      0.159      1.254      0.218      -0.123       0.522
const          1.0897      0.468      2.329      0.025       0.142       2.037
==============================================================================
Omnibus:                        0.070   Durbin-Watson:                   1.853
Prob(Omnibus):                  0.966   Jarque-Bera (JB):                0.093
Skew:                          -0.072   Prob(JB):                        0.955
Kurtosis:                       2.820   Cond. No.                         375.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

# Doing the same for the log database

log_3months_adf_ols = adfuller(eur_train['log'], maxlag=None, autolag="AIC", store=True, regresults=True)[-1].resols.summary()
log_3months_adf_ols

# Results for log values:
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.183
Model:                            OLS   Adj. R-squared:                  0.118
Method:                 Least Squares   F-statistic:                     2.830
Date:                Sun, 29 Sep 2024   Prob (F-statistic):             0.0512
Time:                        19:10:45   Log-Likelihood:                 143.65
No. Observations:                  42   AIC:                            -279.3
Df Residuals:                      38   BIC:                            -272.3
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1            -0.1794      0.077     -2.318      0.026      -0.336      -0.023
x2             0.2473      0.155      1.599      0.118      -0.066       0.560
x3             0.1969      0.159      1.241      0.222      -0.124       0.518
const          0.3232      0.139      2.325      0.025       0.042       0.605
==============================================================================
Omnibus:                        0.081   Durbin-Watson:                   1.860
Prob(Omnibus):                  0.960   Jarque-Bera (JB):                0.106
Skew:                          -0.082   Prob(JB):                        0.948
Kurtosis:                       2.816   Cond. No.                         287.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

# We can notice that in the second model, both AIC and BIC are lower than the first one
# and also that the loglike is higher on the second model than the other, proving
# that the logarithmic version of the data is a better fit for the model so far.

# Getting values from the second model:

log_1year_adf = adfuller(eur_train["log"], maxlag=None, autolag="AIC")
log_1year_adf

# p-value is > 0.05 (0.16), DO NOT REJECT the null hypothesis and suggests the presence of a unit root (NON-STATIONARY)
# adf stats > 10% (-2.31), DO NOT REJECT the null hypothesis and suggests the presence of a unit root (NON-STATIONARY)
# lags = 3 good amount of lags

"""
(-2.3176514907766315,
 0.16633819536001188,
 2,
 42,
 {'1%': -3.596635636000432,
  '5%': -2.933297331821618,
  '10%': -2.6049909750566895},
 -230.1984280656552)
"""

# In[1.2]: Running KPSS test to determine stationarity for eur

log_1year_kpss = kpss(eur_train["log"], regression="c", nlags="auto")
log_1year_kpss

# p-value < 0.05 (0.02) REJECTS the null hypothesis, suggesting data is NON-STATIONARY
# kpss stat of 0.58 is > than 5% (0.463), REJECTS the null hypothesis, suggesting data is NON-STATIONARY
# lags = 10 good amount of lags

"""
(0.5880306853750752,
 0.023724483147720434,
 4,
 {'10%': 0.347, '5%': 0.463, '2.5%': 0.574, '1%': 0.739})
"""

# In[1.3]: Plotting ACF and PACF to examine autocorrelations in data:

plt.style.use("seaborn-v0_8-colorblind")
fig, (ax1, ax2) = plt.subplots(2, figsize=(11, 7), dpi=300, sharex=True)
plot_acf(eur_train["log"], ax=ax1)
plot_pacf(eur_train["log"], ax=ax2)
plt.show()

# PACF plot shows an AR(2) order for the dataset showing a high statistical significance spike 
# at the first lag in PACF. ACF shows slow decay towards 0 -> exponentially decaying or sinusoidal

# In[1.4]: Defining the order of differencing we need:

# Original Series

plt.style.use("seaborn-v0_8-colorblind")
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(11, 8), dpi=300, sharex=True)
ax1.plot(eur_train["log"])
ax1.set_title("Com log")
ax2.plot(eur_train["log"].diff().dropna(), color="green")
ax2.set_title("Com Log e Diferenciação de 1ª ordem")
ax3.plot(eur_train["log"].diff().diff().dropna())
ax3.set_title("Com Log e Diferenciação de 2ª ordem")
plt.suptitle("Comparação dos gráficos de Log com e sem diferenciação", fontsize="18")
plt.show()


# Plotting the ACF for each order

plt.style.use("seaborn-v0_8-colorblind")
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(11, 8), dpi=300, sharex=True)
plot_acf(eur_train["log"], ax=ax1)
ax1.set_title("Com Log")
plot_acf(eur_train["log"].diff().dropna(), ax=ax2, color="green")
ax2.set_title("Com Log e Diferenciação de 1ª ordem")
plot_acf(eur_train["log"].diff().diff().dropna(), ax=ax3)
ax3.set_title("Com Log e Diferenciação de 2ª ordem")
plt.suptitle("Comparação da diferenciação na Autocorrelação (ACF)", fontsize="18")
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

plt.style.use("seaborn-v0_8-colorblind")
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(11, 8), dpi=300, sharex=True)
plot_pacf(eur_train["log"], ax=ax1)
ax1.set_title("Com Log")
plot_pacf(eur_train["log"].diff().dropna(), ax=ax2, color="green")
ax2.set_title("Com Log e Diferenciação de 1ª ordem")
plot_pacf(eur_train["log"].diff().diff().dropna(), ax=ax3)
ax3.set_title("Com Log e Diferenciação de 2ª ordem")
plt.suptitle("Comparação da diferenciação na Autocorrelação Parcial (PACF)", fontsize="18")
plt.show()

# Plotting ACF and PACF together:

fig, (ax1, ax2) = plt.subplots(2, figsize=(11,7), dpi=300)
plot_acf(eur_train["log"].diff().dropna(), ax=ax1)
plot_pacf(eur_train["log"].diff().dropna(), ax=ax2)
plt.show()

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
sns.lineplot(eur_train["diff"], color="limegreen", label="Log(EUR) - Diff")
plt.axhline(y=np.mean(eur_train["diff"]), color="black", linestyle="--", label="Média") # mean for eur
plt.title(f'Log do Euro Diferenciado - Série histórica ({eur_train.index[0].strftime(dt_format)} - {eur_train.index[-1].strftime(dt_format)})', fontsize="18")
plt.yticks(np.arange(round(eur_train["diff"].min(), 2), round(eur_train["diff"].max() + 0.01, 2), 0.01), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b, %Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=29))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Log(Câmbio EUR ↔ BRL) Diferenciado ", fontsize="18")
plt.legend(fontsize=18, loc="lower center")
plt.show()

# In[2.0]: Defining Stationarity again

optimal_lags_eur_1year_diff = 12*(len(eur_train['diff'].dropna())/100)**(1/4)
optimal_lags_eur_1year_diff

# 9.82843510575264

# So optimal lags for this dataset = 14 or 15

# In[2.1]: Augmented Dickey-Fuller Test with diff

log_1year_diff_adf = adfuller(eur_train["diff"].dropna(), maxlag=None, autolag="AIC")
log_1year_diff_adf

# p-value is <<< 0.05 (5.988728689379452e-20), REJECTS the null hypothesis and suggests the absence of a unit root (STATIONARY)
# adf stats <<< 1% (-11), REJECTS the null hypothesis and suggests the absence of a unit root (STATIONARY)
# lags = 2 good amount of lags

"""
(-5.428214168826238,
 2.9543996999618626e-06,
 0,
 44,
 {'1%': -3.5885733964124715,
  '5%': -2.929885661157025,
  '10%': -2.6031845661157025},
 -106.48743912453253)
"""

log_1year_diff_adf_ols = adfuller(eur_train['diff'].dropna(), maxlag=None, autolag="AIC", store=True, regresults=True)[-1].resols.summary()
log_1year_diff_adf_ols

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.412
Model:                            OLS   Adj. R-squared:                  0.398
Method:                 Least Squares   F-statistic:                     29.47
Date:                Sun, 29 Sep 2024   Prob (F-statistic):           2.63e-06
Time:                        19:18:37   Log-Likelihood:                 70.708
No. Observations:                  44   AIC:                            -137.4
Df Residuals:                      42   BIC:                            -133.8
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1            -0.8691      0.160     -5.428      0.000      -1.192      -0.546
const          0.0050      0.008      0.663      0.511      -0.010       0.020
==============================================================================
Omnibus:                        0.583   Durbin-Watson:                   1.918
Prob(Omnibus):                  0.747   Jarque-Bera (JB):                0.158
Skew:                          -0.129   Prob(JB):                        0.924
Kurtosis:                       3.142   Cond. No.                         21.4
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

# AIC, BIC and Loglike prove this model is even better than the second one, so
# I SHOULD USE D = 1 IN THE ARIMA MODEL

# In[2.2]: Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test post differencing

log_1year_diff_kpss = kpss(eur_train['diff'].dropna(), regression="c", nlags="auto")
log_1year_diff_kpss

# p-value > 0.05 (0.1) DO NOT REJECT the null hypothesis, suggesting data is STATIONARY
# kpss stat of 0.06 is < 10% (0.347), DO NOT REJECT the null hypothesis, suggesting data is STATIONARY
# lags = 8 ok amount of lags for this point in time

"""
(0.06718476247232598,
 0.1,
 1,
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

fig, (ax1, ax2) = plt.subplots(2, figsize=(11, 8), dpi=300)
plot_acf(eur_train["diff"].dropna(), ax=ax1)
plot_pacf(eur_train["diff"].dropna(), ax=ax2)
plt.show()

# Both tests seem to show normal stationary plots with no significant lags after zero.
# Also, no significant negative first lag;

# In[3.0]: Running Granger Causality tests

# In[3.1]: Create the the lagged dataframe:

eur_3months_lagged = pd.DataFrame({"log": eur_train["log"]})

eur_3months_lagged['lag 1'] = eur_3months_lagged["log"].shift(1)
eur_3months_lagged['lag 2'] = eur_3months_lagged["log"].shift(2)
eur_3months_lagged['lag 3'] = eur_3months_lagged["log"].shift(3)
eur_3months_lagged['lag 4'] = eur_3months_lagged["log"].shift(4)
eur_3months_lagged['lag 5'] = eur_3months_lagged["log"].shift(5)
eur_3months_lagged['lag 6'] = eur_3months_lagged["log"].shift(6)
eur_3months_lagged['lag 7'] = eur_3months_lagged["log"].shift(7)

eur_3months_lagged

# Running only one lag as this was what adf showed was optimal

# In[3.2]: Running Multicolinearity tests just in case:

eur_constants = add_constant(eur_3months_lagged.dropna())

eur_vif = pd.DataFrame()

eur_vif['vif'] = [variance_inflation_factor(eur_constants.values, i) for i in range(eur_constants.shape[1])]

eur_vif['variable'] = eur_constants.columns

eur_vif

"""
            vif variable
0  18461.536388    const
1      5.861951      log
2     12.096541    lag 1
3     11.646157    lag 2
4     13.389796    lag 3
5     14.625329    lag 4
6     15.161576    lag 5
7     16.885451    lag 6
8      7.357469    lag 7
"""

# We can see very high VIFs between the lags; Which means there's multicolinearity

# In[3.3]: Running the actual Causality test on the lagged data

eur_granger_lag2 = grangercausalitytests(eur_3months_lagged[["log", "lag 2"]].dropna(), maxlag=4)

"""
Granger Causality
number of lags (no zero) 1
ssr based F test:         F=5.1570  , p=0.0288  , df_denom=39, df_num=1
ssr based chi2 test:   chi2=5.5537  , p=0.0184  , df=1
likelihood ratio test: chi2=5.2160  , p=0.0224  , df=1
parameter F test:         F=5.1570  , p=0.0288  , df_denom=39, df_num=1
"""

# Choosing "lag 2" as it's the only dataset with a significant lag.

# In[3.4]: As an experiment, running the EUR against USD:

usd_1year_granger = pd.read_csv("./datasets/wrangled/df_usd_3months.csv", float_precision="high", parse_dates=([0]))

df_eur_usd = pd.DataFrame({"eur": df_eur_3months['log'], "usd": usd_1year_granger["log"]})

eur_usd_granger = grangercausalitytests(df_eur_usd[['eur', 'usd']].dropna(), maxlag=14)

# None of the lags show any statistical p-values for any of the tests.

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


# No significant causality between the two datasets

# In[4.0]: Save final dataset for testing ARIMA

eur_train_3months = pd.DataFrame({
    "date": eur_train.index,
    "eur": eur_train["eur"],
    "log": eur_train["log"],
    "diff": eur_train["diff"],
    "lag2": eur_3months_lagged["lag 2"]
    })

eur_test_3months = pd.DataFrame({
    "date": eur_test.index,
    "eur": eur_test["eur"],
    "log": eur_test["log"],
    "diff": eur_test["diff"]
    })

# save to csv
eur_train_3months.to_csv("./datasets/arima_ready/eur_train_3months.csv", index=False)
eur_test_3months.to_csv("./datasets/arima_ready/eur_test_3months.csv", index=False)
