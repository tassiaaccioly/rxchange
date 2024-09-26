# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # #
# Running the ARIMA model = EURO - 1 year #
# # # # # # # # # # # # # # # # # # # # # #

# In[0.1]: Importação dos pacotes

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import norm, shapiro, jarque_bera
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import mean, stdev

# In[0.2]: Define important functions

# Function to invert the result value (inverse diff then inverse log)
def invert_results(series_diff, first_value):
    print(series_diff)
    print(first_value)
    series_inverted = np.r_[first_value, series_diff].cumsum().astype('float64')
    return np.exp(series_inverted)

# test_difffunc = pd.DataFrame({"eur": invert_results(df_eur_arima_train["diff"], eur_arima_1year.iloc[5,2])})

# Normal distribution:

nd = np.arange(-0.015, 0.020, 0.001)

mn = mean(nd)
sd = stdev(nd)


# In[0.2]: Import dataframes

eur_arima_1year = pd.read_csv("./datasets/arima_ready/eur_arima_1year.csv", float_precision="high", parse_dates=([0]))

df_eur_arima_train = pd.read_csv("./datasets/arima_ready/eur_arima_1year.csv", float_precision="high", parse_dates=([0])).dropna()

df_eur_arima_train.info()

df_eur_arima_test = pd.read_csv("./datasets/arima_ready/eur_arima_3months.csv", float_precision="high", parse_dates=([0]))

df_eur_arima_test.info()

y = df_eur_arima_train["diff"]

test_diff = df_eur_arima_test["log"]

# In[1.0]: Training the model SARIMAX(0,0,2) + intercept

eur_arima_final_002int = SARIMAX(y, exog=None, order=(0,0,2), enforce_stationarity=True, trend="c").fit()

eur_arima_final_002int.summary()

"""
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                   diff   No. Observations:                  274
Model:               SARIMAX(0, 0, 2)   Log Likelihood                1066.934
Date:                Thu, 26 Sep 2024   AIC                          -2125.869
Time:                        14:27:38   BIC                          -2111.416
Sample:                             0   HQIC                         -2120.068
                                - 274                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
intercept      0.0004      0.000      1.614      0.106   -8.34e-05       0.001
ma.L1         -0.0935      0.058     -1.607      0.108      -0.207       0.021
ma.L2         -0.1238      0.062     -1.983      0.047      -0.246      -0.001
sigma2      2.424e-05   1.48e-06     16.378      0.000    2.13e-05    2.71e-05
===================================================================================
Ljung-Box (L1) (Q):                   0.00   Jarque-Bera (JB):                62.41
Prob(Q):                              0.95   Prob(JB):                         0.00
Heteroskedasticity (H):               1.09   Skew:                             0.41
Prob(H) (two-sided):                  0.68   Kurtosis:                         5.19
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
"""

eur_arima_final_002int.plot_diagnostics(figsize=(10,7))
plt.show()



# In[1.1]: Saving and plotting the residuals for inspection

df_eur_arima_train["erros002_int"] = eur_arima_final_002int.resid

erros002_int = df_eur_arima_train["erros002_int"].dropna()

erros002_int.describe()


# Plot ACF and PACF of Residuals:

plt.rcParams.update({'figure.figsize':(11,8), 'figure.dpi':120})

fig, (ax1, ax2) = plt.subplots(2)
plot_acf(erros002_int, ax=ax1)
plot_pacf(erros002_int, ax=ax2)

# Plot the scatter plot

plt.figure(figsize=(15, 10))
sns.scatterplot(x=df_eur_arima_train["date"], y=erros002_int, color="green", alpha=0.4,
             edgecolor=None)
plt.title("Resíduos do Modelo - ARIMA(0,0,2) + intercepto", fontsize="18")
plt.xlabel("")
plt.ylabel("")
plt.legend(fontsize=18, loc="upper right")
plt.show()


# In[1.2]: Run Residuals tests:

## Ljung-Box test

acorr_ljungbox(erros002_int, lags=[1, 7, 14, 21, 30], return_df=True)

"""
   lb_stat  lb_pvalue
1  0.01033   0.919046
"""

# We can see that the p-value is much higher than 0.05, so we DO NOT REJECT the
# null hypothesis, indicating there's no autocorrelation.

# Testing with more lags:

acorr_ljungbox(erros002_int, lags=[7], return_df=True)

"""
   lb_stat  lb_pvalue
7  5.87906    0.55394
"""

acorr_ljungbox(erros002_int, lags=[14], return_df=True)

"""
      lb_stat  lb_pvalue
14  11.721758   0.628637
"""

# Again, we see values much higher than 0.05, so we DO NOT REJECT the null hypothesis
# which indicates that the residuals have no autocorrelation between them.

# Also running for the 30 day mark:

acorr_ljungbox(erros002_int, lags=[30], return_df=True)

"""
      lb_stat  lb_pvalue
30  23.699965   0.785426
"""

# Again we see no autocorrelation at the 30 days mark.

## SHAPIRO-WILK TEST

shapiro(erros002_int)

# Shapiro Result
#            statistic |                pvalue
#    0.9511906745606377 | 8.299140554001586e-08

# p-value < 0.05 so we REJECT the null hypothesis, so we cannot confirm residuals
# are normally distribuited
# Shapiro-Wilk shows that the residuals are not normally distributed.

## JARQUE-BERA TEST

jarque_bera(erros002_int)

# Significance Result
#           statistic |                 pvalue
#  59.480929770497724 | 1.2130542146839541e-13

# The p-value < 0.05 so we REJECT the null hypothesis and confirm residuals DO
# NOT FOLLOW A NORMAL DISTRIBUTION
# The Jarque-Bera test comproves the residuals are NOT NORMALLY DISTRIBUTED

# In[2.0]: Training the model SARIMAX(0,0,0)

eur_arima_final_000 = SARIMAX(y, exog=None, order=(0,0,0)).fit()

eur_arima_final_000.summary()

"""
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                   diff   No. Observations:                  274
Model:                        SARIMAX   Log Likelihood                1062.944
Date:                Thu, 26 Sep 2024   AIC                          -2123.888
Time:                        13:43:32   BIC                          -2120.275
Sample:                             0   HQIC                         -2122.438
                                - 274                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
sigma2        2.5e-05   1.46e-06     17.081      0.000    2.21e-05    2.79e-05
===================================================================================
Ljung-Box (L1) (Q):                   1.25   Jarque-Bera (JB):                59.09
Prob(Q):                              0.26   Prob(JB):                         0.00
Heteroskedasticity (H):               1.13   Skew:                             0.31
Prob(H) (two-sided):                  0.57   Kurtosis:                         5.19
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
"""

eur_arima_final_000.plot_diagnostics(figsize=(10,7))
plt.show()


# In[2.1]: Saving and plotting the residuals for inspection

df_eur_arima_train["erros000"] = eur_arima_final_000.resid

erros000 = df_eur_arima_train["erros000"].dropna()


# Plot ACF and PACF of Residuals:

plt.rcParams.update({'figure.figsize':(11,8), 'figure.dpi':120})

fig, (ax1, ax2) = plt.subplots(2)
plot_acf(erros000, ax=ax1)
plot_pacf(erros000, ax=ax2)

# Plot the scatter plot

plt.figure(figsize=(15, 10))
sns.scatterplot(x=df_eur_arima_train["date"], y=erros000, color="green", alpha=0.4,
             edgecolor=None)
plt.title("Resíduos do Modelo - ARIMA(0,0,2) + intercepto", fontsize="18")
plt.xlabel("")
plt.ylabel("")
plt.legend(fontsize=18, loc="upper right")
plt.show()

# In[1.2]: Run Residuals tests:

## Ljung-Box test

acorr_ljungbox(erros000, lags=[1, 7, 14, 21, 30], return_df=True)

"""
      lb_stat  lb_pvalue
1    1.144517   0.284700
7    9.394904   0.225533
14  14.967923   0.380351
21  19.062914   0.581104
30  28.392936   0.549623
"""

# We can see that the p-value is much higher than 0.05 for all lags, so we DO
# NOT REJECT the null hypothesis, indicating there's no autocorrelation in the
# residuals

## SHAPIRO-WILK TEST

shapiro(erros000)

# Shapiro Result
#             statistic |                 pvalue
#    0.9474999916715173 | 3.2575314671606276e-08

# p-value < 0.05 so we REJECT the null hypothesis, so we cannot confirm residuals
# are normally distribuited
# Shapiro-Wilk shows that the residuals are not normally distributed.

## JARQUE-BERA TEST

jarque_bera(erros000)

# Significance Result
#          statistic |                pvalue
#  55.54183061957464 | 8.694502237054151e-13

# The p-value < 0.05 so we REJECT the null hypothesis and confirm residuals DO
# NOT FOLLOW A NORMAL DISTRIBUTION
# The Jarque-Bera test comproves the residuals are NOT NORMALLY DISTRIBUTED

# In[]: Predict

# testing the model

eur_arima_test_final = eur_arima_final_002.apply(df_eur_arima_test["diff"])

# Fazendo previsões:

eur_arima_002_predict = eur_arima_final_002.predict()

# In[]: Testing the fitted values

df_eur_arima_train["yhat"] = eur_arima_final.fittedvalues
eur_arima_fitted = pd.DataFrame(eur_arima_final.fittedvalues)

# In[]: 

# Plot the histogram

mu, std = norm.fit(eur_arima_final_002.resid)


plt.figure(figsize=(15, 10))
sns.histplot(x=eur_arima_final_002.resid, color="green", alpha=0.4,
             edgecolor=None, kde=True, line_kws={
                 "linewidth": 3, "linestyle": "dashed", "color": "m",
                 "label": "KDE"}, 
             )
xmin, xmax = plt.xlim()
x = np.linspace(-0.015, 0.020, len(eur_arima_final_002.resid))
p = norm.pdf(x, mu, std)
plt.plot(x, p, "k", color="magenta", linewidth=3, label="Distribuição Normal")
plt.title("Resíduos do Modelo - ARIMA(0,0,2) + intercepto", fontsize="18")
plt.xlabel("")
plt.ylabel("")
plt.legend(fontsize=18, loc="upper right")
plt.show()

# Plot the scatter plot

plt.figure(figsize=(15, 10))
sns.scatterplot(x=df_eur_arima_train["date"], y=df_eur_arima_train["erros"], color="green", alpha=0.4,
             edgecolor=None)
plt.title("Resíduos do Modelo - ARIMA(0,0,2) + intercepto", fontsize="18")
plt.xlabel("")
plt.ylabel("")
plt.legend(fontsize=18, loc="upper right")
plt.show()