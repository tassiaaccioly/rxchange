# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # #
# Running the ARIMA model = EURO - 1 year #
# # # # # # # # # # # # # # # # # # # # # #

# In[0.1]: Importação dos pacotes

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


# In[0.2]: Import dataframes

df_eur_arima_train = pd.read_csv("./datasets/arima_ready/eur_arima_1year.csv", float_precision="high", parse_dates=([0]))

df_eur_arima_train.info()

df_eur_arima_test = pd.read_csv("./datasets/arima_ready/eur_arima_3months.csv", float_precision="high", parse_dates=([0]))

df_eur_arima_test.info()

# In[1.0]: Training the model

eur_arima_final = ARIMA(df_eur_arima_train['diff'], exog=None, order=(0,0,2), enforce_stationarity=True).fit()

eur_arima_final.summary()

"""
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                   diff   No. Observations:                  274
Model:                 ARIMA(0, 0, 2)   Log Likelihood                1066.934
Date:                Tue, 24 Sep 2024   AIC                          -2125.868
Time:                        20:42:40   BIC                          -2111.416
Sample:                             0   HQIC                         -2120.067
                                - 274                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const          0.0004      0.000      1.625      0.104   -8.08e-05       0.001
ma.L1         -0.0937      0.058     -1.610      0.107      -0.208       0.020
ma.L2         -0.1235      0.063     -1.975      0.048      -0.246      -0.001
sigma2      2.427e-05   1.48e-06     16.357      0.000    2.14e-05    2.72e-05
===================================================================================
Ljung-Box (L1) (Q):                   0.00   Jarque-Bera (JB):                62.43
Prob(Q):                              0.94   Prob(JB):                         0.00
Heteroskedasticity (H):               1.09   Skew:                             0.41
Prob(H) (two-sided):                  0.68   Kurtosis:                         5.19
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
"""

eur_arima_results = ARIMAResults(eur_arima_final)

# testing the model

eur_arima_final.apply(df_eur_arima_test["diff"])

# Fazendo previsões:

eur_arima_predict = eur_arima_final.predict()

# Function to invert the value
def diff_inv(series_diff, first_value):
    series_inverted = np.r_[first_value, series_diff].cumsum().astype('float64')
    return series_inverted


# In[1.1]: Saving and plotting the residuals

eur_arima_resid = pd.DataFrame(eur_arima_final.resid)
eur_arima_fitted = pd.DataFrame(eur_arima_final.fittedvalues)

df_eur_arima_train["erros"] = eur_arima_resid
df_eur_arima_train["yhat"] = eur_arima_final.fittedvalues

eur_arima_resid.plot(label="Residuals")

plt.figure(figsize=(15, 10))
sns.histplot(x=df_eur_arima_train["erros"], color="green", alpha=0.4,
             edgecolor=None, kde=True, line_kws={
                 "linewidth": 3, "linestyle": "dashed", "color": "m"
                 }
             )
plt.title("Resíduos do Modelo - ARIMA(0,0,2) + intercepto", fontsize="18")
plt.xlabel("")
plt.ylabel("")
plt.legend(fontsize=18, loc="upper right")
plt.show()


