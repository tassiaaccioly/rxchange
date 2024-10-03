# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # #
# Exploratory Data Analysis = USDO - 365 days #
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

df_usd_3months = pd.read_csv("./datasets/wrangled/df_usd_3months.csv", float_precision="high", parse_dates=([1]))

df_usd_3months.info()

df_wg_usd_3months = df_usd_3months

dt_format = "%d/%m/%Y"

df_wg_usd_3months = df_wg_usd_3months.drop(df_wg_usd_3months[df_wg_usd_3months["weekday"] == "Sunday"].index)

df_wg_usd_3months = df_wg_usd_3months.drop(df_wg_usd_3months[df_wg_usd_3months["weekday"] == "Saturday"].index)[1:]

df_wg_usd_3months = df_wg_usd_3months.drop(["weekday", "date"], axis=1)

df_wg_usd_3months = pd.DataFrame(df_wg_usd_3months.drop("dateTime", axis=1).values, index=pd.date_range(start="2024-06-24", periods=65 ,freq="B"), columns=["usd", "log", "diff"])


# In[0.3]: Separate data into train and test

# remove last 15 days of data for test later
usd_train = df_wg_usd_3months[:-20]

usd_test = df_wg_usd_3months[-20:]

# In[0.4]: Plot original datasets

# 3 months dataset
sns.set_style("whitegrid",  {"grid.linestyle": ":"})
plt.figure(figsize=(15, 10), dpi=300)
sns.lineplot(usd_train["usd"], color="limegreen", label="Câmbio USD")
plt.axhline(y=np.mean(usd_train["usd"]), color="black", linestyle="--", label="Média", linewidth=2) # mean for usdo
plt.axhline(y=np.max(usd_train["usd"]), color="magenta", label="Máxima", linewidth=2) # máxima for usdo
plt.axhline(y=np.min(usd_train["usd"]), color="magenta", linestyle="--", label="Mínima", linewidth=2) # máxima for usdo
plt.title(f'Cotação do Euro - Série histórica ({df_usd_3months["dateTime"][0].strftime(dt_format)} - {df_usd_3months["dateTime"][45].strftime(dt_format)})', fontsize="18")
plt.yticks(np.arange(round(usd_train["usd"].min(), 1), round(usd_train["usd"].max() + 0.1, 1), 0.05), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d, %b'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Câmbio USD ↔ BRL", fontsize="18")
plt.legend(fontsize=18, loc="upper left", bbox_to_anchor=(0.02, 0, 0, 0.93))
plt.show()

# Boxplot for 3months:

plt.figure(figsize=(15,10), dpi=300)
sns.boxplot(x=usd_train["usd"], palette="viridis")
plt.title("Boxplot do Dataset USD - 3 meses - Valor Real", fontsize="18")
plt.legend(fontsize="16")
plt.show()


plt.figure(figsize=(15,10), dpi=300)
sns.boxplot(x=usd_train["diff"], palette="viridis")
plt.title("Boxplot do Dataset USD - 3 meses - Diff", fontsize="18")
plt.legend(fontsize="16")
plt.show()

# In[0.5]: Calculate Statistics for datasets

var_usd_3months = np.var(usd_train['usd'])
var_usd_3months

# 0.009373995424691363

varlog_usd_3months = np.var(usd_train['log'])
varlog_usd_3months

# 0.00030331188535171095

optimal_lags_usd_3months = 12*(len(usd_train['usd'])/100)**(1/4)
optimal_lags_usd_3months

# 9.82843510575264

# So optimal lags for this dataset = 9 or 10

# In[0.6]: Plot train and test datasets:

plt.figure(figsize=(15, 10), dpi=300)
sns.lineplot(usd_train["log"], color="limegreen", label="train log(Câmbio USD)")
sns.lineplot(usd_test["log"], color="magenta", label="test log(Câmbio USD)")
plt.axhline(y=np.mean(usd_train["log"]), color="black", linestyle="--", label="Média") # mean for usdo
plt.axhline(y=np.max(usd_train["log"]), color="magenta", label="Máxima") # máxima for usdo
plt.axhline(y=np.min(usd_train["log"]), color="magenta", linestyle="--", label="Mínima") # máxima for usdo
plt.axvline(x=usd_test.index[0], color="green", linestyle="--", linewidth="3")
plt.title(f'Cotação do Euro - Série histórica ({usd_train.index[0].strftime(dt_format)} - {usd_test.index[-1].strftime(dt_format)})', fontsize="18")
plt.yticks(np.arange(round(df_wg_usd_3months["log"].min(), 2), round(df_wg_usd_3months["log"].max() + 0.01, 2), 0.01), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d, %b'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Câmbio log(USD ↔ BRL)", fontsize="18")
plt.legend(fontsize=18, loc="center", bbox_to_anchor=(1.16, 0.5, 0, 0))
plt.show()

# In[0.7]: Doing the seasonal decompose


usd_decom_multi = seasonal_decompose(usd_train["usd"], model="multiplicative")
plt.figure(figsize=(15,10), dpi=300)
usd_decom_multi.plot()
plt.show()

# Plotting season and residuals
plt.figure(figsize=(15,6), dpi=300)
usd_decom_multi.seasonal.plot(legend=True)
usd_decom_multi.resid.plot(legend=True)
plt.xlabel('Períodos',size=15)
plt.title('Sazonalidade e resíduos',size=15);

# plotting trend
plt.figure(figsize=(15,6))
usd_decom_multi.trend.plot(legend=True, color='r')
plt.xlabel('Períodos',size=15)
plt.title('Série de tendência',size=15);

usd_detrended = (usd_decom_multi.seasonal*usd_decom_multi.resid)
plt.figure(figsize=(15,6))
usd_detrended.plot()
plt.xlabel('Períodos',size=15)
plt.title('Gráfico da série detendencionalizada de forma multiplicativa',size=15);

usd_train['trend']=usd_decom_multi.trend
usd_train['detrend']=usd_train['usd']/usd_train['trend']
#gráfico das séries original e dessazonalizada
usd_train[['usd','detrend']].plot(figsize=(15,6))
plt.xlabel('Períodos',size=15)
plt.title('Série de consumo de cerveja com e sem sazonalidade',size=15);

# In[1.0]: Defining Stationarity

# In[1.1]: Augmented Dickey-Fuller Test with usd and log

usd_3months_adf_ols = adfuller(usd_train['usd'], maxlag=None, autolag="AIC", store=True, regresults=True)[-1].resols.summary()
usd_3months_adf_ols

# Normal values results:
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.278
Model:                            OLS   Adj. R-squared:                  0.172
Method:                 Least Squares   F-statistic:                     2.619
Date:                Mon, 30 Sep 2024   Prob (F-statistic):             0.0416
Time:                        21:56:01   Log-Likelihood:                 69.397
No. Observations:                  40   AIC:                            -126.8
Df Residuals:                      34   BIC:                            -116.7
Df Model:                           5                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1            -0.2616      0.092     -2.830      0.008      -0.449      -0.074
x2             0.3282      0.163      2.018      0.052      -0.002       0.659
x3             0.2824      0.163      1.728      0.093      -0.050       0.615
x4            -0.0931      0.169     -0.553      0.584      -0.436       0.249
x5             0.3120      0.170      1.840      0.074      -0.033       0.657
const          1.4533      0.513      2.834      0.008       0.411       2.495
==============================================================================
Omnibus:                        0.551   Durbin-Watson:                   1.671
Prob(Omnibus):                  0.759   Jarque-Bera (JB):                0.398
Skew:                           0.237   Prob(JB):                        0.820
Kurtosis:                       2.882   Cond. No.                         408.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

# Doing the same for the log database

log_3months_adf_ols = adfuller(usd_train['log'], maxlag=None, autolag="AIC", store=True, regresults=True)[-1].resols.summary()
log_3months_adf_ols

# Results for log values:
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.277
Model:                            OLS   Adj. R-squared:                  0.170
Method:                 Least Squares   F-statistic:                     2.602
Date:                Mon, 30 Sep 2024   Prob (F-statistic):             0.0426
Time:                        21:56:20   Log-Likelihood:                 138.04
No. Observations:                  40   AIC:                            -264.1
Df Residuals:                      34   BIC:                            -253.9
Df Model:                           5                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1            -0.2623      0.092     -2.837      0.008      -0.450      -0.074
x2             0.3228      0.163      1.985      0.055      -0.008       0.653
x3             0.2861      0.163      1.751      0.089      -0.046       0.618
x4            -0.0868      0.169     -0.515      0.610      -0.429       0.256
x5             0.3106      0.170      1.829      0.076      -0.035       0.656
const          0.4498      0.158      2.840      0.008       0.128       0.772
==============================================================================
Omnibus:                        0.427   Durbin-Watson:                   1.672
Prob(Omnibus):                  0.808   Jarque-Bera (JB):                0.337
Skew:                           0.211   Prob(JB):                        0.845
Kurtosis:                       2.847   Cond. No.                         328.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

# We can notice that in the second model, both AIC and BIC are lower than the first one
# and also that the loglike is higher on the second model than the other, proving
# that the logarithmic version of the data is a better fit for the model so far.

# Getting values from the second model:

log_1year_adf = adfuller(usd_train["log"], maxlag=None, autolag="AIC")
log_1year_adf

# p-value is > 0.05 (0.0531), DO NOT REJECT the null hypothesis and suggests the presence of a unit root (NON-STATIONARY)
# adf stats > 10% (-2.83), DO NOT REJECT the null hypothesis and suggests the presence of a unit root (NON-STATIONARY)
# lags = 3 good amount of lags

"""
(-2.837400774185113,
 0.053126737863115094,
 4,
 40,
 {'1%': -3.6055648906249997, '5%': -2.937069375, '10%': -2.606985625},
 -230.39068910037815)
"""

# In[1.2]: Running KPSS test to determine stationarity for usd

log_1year_kpss = kpss(usd_train["log"], regression="c", nlags="auto")
log_1year_kpss

# p-value > 0.05 (0.1) DO NOT REJECT the null hypothesis, suggesting data is STATIONARY
# kpss stat of 0.16 is < than 5% (0.463), DO NOT REJECT the null hypothesis, suggesting data is STATIONARY
# lags = 10 good amount of lags

"""
(0.16550300828965164,
 0.1,
 4,
 {'10%': 0.347, '5%': 0.463, '2.5%': 0.574, '1%': 0.739})
"""

# In[1.3]: Plotting ACF and PACF to examine autocorrelations in data:

plt.style.use("seaborn-v0_8-colorblind")
fig, (ax1, ax2) = plt.subplots(2, figsize=(11, 7), dpi=300, sharex=True)
plot_acf(usd_train["log"], ax=ax1)
plot_pacf(usd_train["log"], ax=ax2)
plt.show()

# PACF plot shows an AR(2) order for the dataset showing a high statistical significance spike 
# at the first lag in PACF. ACF shows slow decay towards 0 -> exponentially decaying or sinusoidal

# In[1.4]: Defining the order of differencing we need:

# Original Series

plt.style.use("seaborn-v0_8-colorblind")
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(11, 8), dpi=300, sharex=True)
ax1.plot(usd_train["log"])
ax1.set_title("Com log")
ax2.plot(usd_train["log"].diff().dropna(), color="green")
ax2.set_title("Com Log e Diferenciação de 1ª ordem")
ax3.plot(usd_train["log"].diff().diff().dropna())
ax3.set_title("Com Log e Diferenciação de 2ª ordem")
plt.suptitle("Comparação dos gráficos de Log com e sem diferenciação", fontsize="18")
plt.show()


# Plotting the ACF for each order

plt.style.use("seaborn-v0_8-colorblind")
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(11, 8), dpi=300, sharex=True)
plot_acf(usd_train["log"], ax=ax1)
ax1.set_title("Com Log")
plot_acf(usd_train["log"].diff().dropna(), ax=ax2, color="green")
ax2.set_title("Com Log e Diferenciação de 1ª ordem")
plot_acf(usd_train["log"].diff().diff().dropna(), ax=ax3)
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
plot_pacf(usd_train["log"], ax=ax1)
ax1.set_title("Com Log")
plot_pacf(usd_train["log"].diff().dropna(), ax=ax2, color="green")
ax2.set_title("Com Log e Diferenciação de 1ª ordem")
plot_pacf(usd_train["log"].diff().diff().dropna(), ax=ax3)
ax3.set_title("Com Log e Diferenciação de 2ª ordem")
plt.suptitle("Comparação da diferenciação na Autocorrelação Parcial (PACF)", fontsize="18")
plt.show()

# Plotting ACF and PACF together:

fig, (ax1, ax2) = plt.subplots(2, figsize=(11,7), dpi=300)
plot_acf(usd_train["log"].diff().dropna(), ax=ax1)
plot_pacf(usd_train["log"].diff().dropna(), ax=ax2)
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
sns.lineplot(usd_train["diff"], color="limegreen", label="Log(USD) - Diff")
plt.axhline(y=np.mean(usd_train["diff"]), color="black", linestyle="--", label="Média") # mean for usd
plt.title(f'Log do Euro Diferenciado - Série histórica ({usd_train.index[0].strftime(dt_format)} - {usd_train.index[-1].strftime(dt_format)})', fontsize="18")
plt.yticks(np.arange(round(usd_train["diff"].min(), 2), round(usd_train["diff"].max() + 0.01, 2), 0.01), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b, %Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=29))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Log(Câmbio USD ↔ BRL) Diferenciado ", fontsize="18")
plt.legend(fontsize=18, loc="lower center")
plt.show()

# In[2.0]: Defining Stationarity again

optimal_lags_usd_1year_diff = 12*(len(usd_train['diff'].dropna())/100)**(1/4)
optimal_lags_usd_1year_diff

# 9.82843510575264

# So optimal lags for this dataset = 14 or 15

# In[2.1]: Augmented Dickey-Fuller Test with diff

log_1year_diff_adf = adfuller(usd_train["diff"].dropna(), maxlag=None, autolag="AIC")
log_1year_diff_adf

# p-value is <<< 0.05 (3.922368357423254e-06), REJECTS the null hypothesis and suggests the absence of a unit root (STATIONARY)
# adf stats <<< 1% (-11), REJECTS the null hypothesis and suggests the absence of a unit root (STATIONARY)
# lags = 2 good amount of lags

"""
(-5.369605306274183,
 3.922368357423254e-06,
 0,
 44,
 {'1%': -3.5885733964124715,
  '5%': -2.929885661157025,
  '10%': -2.6031845661157025},
 -112.35658305099176)
"""

log_1year_diff_adf_ols = adfuller(usd_train['diff'].dropna(), maxlag=None, autolag="AIC", store=True, regresults=True)[-1].resols.summary()
log_1year_diff_adf_ols

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.407
Model:                            OLS   Adj. R-squared:                  0.393
Method:                 Least Squares   F-statistic:                     28.83
Date:                Mon, 30 Sep 2024   Prob (F-statistic):           3.18e-06
Time:                        22:02:12   Log-Likelihood:                 71.714
No. Observations:                  44   AIC:                            -139.4
Df Residuals:                      42   BIC:                            -135.9
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1            -0.8744      0.163     -5.370      0.000      -1.203      -0.546
const          0.0027      0.007      0.371      0.712      -0.012       0.017
==============================================================================
Omnibus:                        0.340   Durbin-Watson:                   1.846
Prob(Omnibus):                  0.844   Jarque-Bera (JB):                0.030
Skew:                          -0.051   Prob(JB):                        0.985
Kurtosis:                       3.078   Cond. No.                         22.3
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

# AIC, BIC and Loglike prove this model is even better than the second one, so
# I SHOULD USE D = 1 IN THE ARIMA MODEL

# In[2.2]: Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test post differencing

log_1year_diff_kpss = kpss(usd_train['diff'].dropna(), regression="c", nlags="auto")
log_1year_diff_kpss

# p-value > 0.05 (0.1) DO NOT REJECT the null hypothesis, suggesting data is STATIONARY
# kpss stat of 0.06 is < 10% (0.347), DO NOT REJECT the null hypothesis, suggesting data is STATIONARY
# lags = 8 ok amount of lags for this point in time

"""
(0.060248824234724,
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

# In[2.3]: Plotting ACF and PACF to determine correct number of lags for usd

fig, (ax1, ax2) = plt.subplots(2, figsize=(11, 8), dpi=300)
plot_acf(usd_train["diff"].dropna(), ax=ax1)
plot_pacf(usd_train["diff"].dropna(), ax=ax2)
plt.show()

# Both tests seem to show normal stationary plots with no significant lags after zero.
# Also, no significant negative first lag;

# In[3.0]: Running Granger Causality tests

# In[3.1]: Create the the lagged dataframe:

usd_3months_lagged = pd.DataFrame({"log": usd_train["log"]})

usd_3months_lagged['lag 1'] = usd_3months_lagged["log"].shift(1)
usd_3months_lagged['lag 2'] = usd_3months_lagged["log"].shift(2)
usd_3months_lagged['lag 3'] = usd_3months_lagged["log"].shift(3)
usd_3months_lagged['lag 4'] = usd_3months_lagged["log"].shift(4)
usd_3months_lagged['lag 5'] = usd_3months_lagged["log"].shift(5)
usd_3months_lagged['lag 6'] = usd_3months_lagged["log"].shift(6)
usd_3months_lagged['lag 7'] = usd_3months_lagged["log"].shift(7)

usd_3months_lagged

# Running only one lag as this was what adf showed was optimal

# In[3.2]: Running Multicolinearity tests just in case:

usd_constants = add_constant(usd_3months_lagged.dropna())

usd_vif = pd.DataFrame()

usd_vif['vif'] = [variance_inflation_factor(usd_constants.values, i) for i in range(usd_constants.shape[1])]

usd_vif['variable'] = usd_constants.columns

usd_vif

"""
            vif variable
0  21859.200872    const
1      5.759594      log
2     13.239906    lag 1
3     14.439347    lag 2
4     13.711811    lag 3
5     13.853888    lag 4
6     14.518626    lag 5
7     14.160610    lag 6
8      5.846402    lag 7
"""

# We can see very high VIFs between the lags; Which means there's multicolinearity

# In[3.3]: Running the actual Causality test on the lagged data

usd_granger_lag2 = grangercausalitytests(usd_3months_lagged[["log", "lag 7"]].dropna(), maxlag=4)

# lag 2

"""
Granger Causality
number of lags (no zero) 1
ssr based F test:         F=5.2701  , p=0.0272  , df_denom=39, df_num=1
ssr based chi2 test:   chi2=5.6755  , p=0.0172  , df=1
likelihood ratio test: chi2=5.3234  , p=0.0210  , df=1
parameter F test:         F=5.2701  , p=0.0272  , df_denom=39, df_num=1
"""

# lag 3

"""
Granger Causality
number of lags (no zero) 1
ssr based F test:         F=5.2701  , p=0.0272  , df_denom=39, df_num=1
ssr based chi2 test:   chi2=5.6755  , p=0.0172  , df=1
likelihood ratio test: chi2=5.3234  , p=0.0210  , df=1
parameter F test:         F=5.2701  , p=0.0272  , df_denom=39, df_num=1
"""

# lag 4

"""
Granger Causality
number of lags (no zero) 1
ssr based F test:         F=5.9566  , p=0.0196  , df_denom=37, df_num=1
ssr based chi2 test:   chi2=6.4395  , p=0.0112  , df=1
likelihood ratio test: chi2=5.9708  , p=0.0145  , df=1
parameter F test:         F=5.9566  , p=0.0196  , df_denom=37, df_num=1
"""

# Choosing "lag 4" as it has the lowest p-values

# In[3.4]: As an experiment, running the USD against USD:

eur_3months_granger = pd.read_csv("./datasets/wrangled/df_eur_3months.csv", float_precision="high", parse_dates=([0]))

df_usd_usd = pd.DataFrame({"usd": df_usd_3months['log'], "eur": eur_3months_granger["log"]})

usd_usd_granger = grangercausalitytests(df_usd_usd[['usd', 'eur']].dropna(), maxlag=14)

# None of the lags show any statistical p-values for any of the tests.

# In[3.6]: Cross-testing the series to make sure they have causality between them

tscv = TimeSeriesSplit(n_splits=5)

ct_usd_usd = df_usd_usd.dropna()

for train_index, test_index in tscv.split(ct_usd_usd):
    train, test = ct_usd_usd.iloc[train_index], ct_usd_usd.iloc[test_index]
    
    X_train, y_train = train['eur'], train['usd']
    X_test, y_test = test['eur'], test['usd']
    
    granger_result = grangercausalitytests(train[['eur', 'usd']], maxlag=4, verbose=False)
    
    for lag, result in granger_result.items():
        f_test_pvalue = result[0]['ssr_ftest'][1]  # p-value from F-test
        print(f"Lag: {lag}, P-value: {f_test_pvalue}")
    
    print(f"TRAIN indices: {train_index}")
    print(f"TEST indices: {test_index}")


# No significant causality between the two datasets

# In[4.0]: Save final dataset for testing ARIMA

usd_train_3months = pd.DataFrame({
    "date": usd_train.index,
    "usd": usd_train["usd"],
    "log": usd_train["log"],
    "diff": usd_train["diff"],
    "lag": usd_3months_lagged["lag 4"]
    })

usd_test_3months = pd.DataFrame({
    "date": usd_test.index,
    "usd": usd_test["usd"],
    "log": usd_test["log"],
    "diff": usd_test["diff"]
    })

# save to csv
usd_train_3months.to_csv("./datasets/arima_ready/usd_train_3months.csv", index=False)
usd_test_3months.to_csv("./datasets/arima_ready/usd_test_3months.csv", index=False)
