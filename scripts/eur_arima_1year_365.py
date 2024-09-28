# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # #
# Finding the ARIMA model = EURO - 1 year #
# # # # # # # # # # # # # # # # # # # # # #

# In[0.1]: Importação dos pacotes

import pandas as pd
from pmdarima import auto_arima
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.api import SARIMAX


# In[0.2]: Import dataframes

eur_train_1year = pd.read_csv("./datasets/arima_ready/eur_train_1year.csv", float_precision="high", parse_dates=([0]))
 
eur_arima_train = eur_train_1year

eur_arima_train.info()

eur_test_1year = pd.read_csv("./datasets/arima_ready/eur_test_1year.csv", float_precision="high", parse_dates=([0]))
 
eur_arima_test = eur_test_1year

eur_arima_test.info()

# In[0.3]: Creating exogs

# Parameters:
# AR - Up to 2
# d = 1
# MA - Up to 2

eur_exog1 = pd.concat([eur_arima_train["lag7"].shift(1), eur_arima_train["lag7"].shift(3)], axis=1).dropna()
eur_exog2 = pd.concat([eur_arima_train["lag7"].diff().shift(1), eur_arima_train["lag7"].diff().shift(3)], axis=1).dropna()

len_eur_original = len(eur_arima_train)
len_eur_exog1 = len_eur_original - len(eur_exog1) - 1
len_eur_exog2 = len_eur_original - len(eur_exog2) - 1

df_eur_exog1 = eur_arima_train['eur'].drop(eur_arima_train['eur'].loc[0:len_eur_exog1].index)
df_eur_exog2 = eur_arima_train['diff'].drop(eur_arima_train['diff'].loc[0:len_eur_exog2].index)

# In[1.0]: Run automated ARIMA

eur_train_AIC_fit = auto_arima(y = eur_arima_train['eur'],
                               test="adf",
                               m=1,
                               seasonal = False,
                               stepwise = True,
                               trace = True
                               ).fit(y = eur_arima_train['eur'])

eur_train_AIC_fit.summary()

"""
Performing stepwise search to minimize aic
 ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=-1171.444, Time=0.48 sec
 ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=-1171.014, Time=0.02 sec
 ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=-1169.171, Time=0.02 sec
 ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=-1169.161, Time=0.09 sec
 ARIMA(0,1,0)(0,0,0)[0]             : AIC=-1170.295, Time=0.01 sec
 ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=-1169.094, Time=0.35 sec
 ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=-1169.003, Time=0.41 sec
 ARIMA(3,1,2)(0,0,0)[0] intercept   : AIC=-1173.175, Time=0.49 sec
 ARIMA(3,1,1)(0,0,0)[0] intercept   : AIC=-1171.279, Time=0.32 sec
 ARIMA(4,1,2)(0,0,0)[0] intercept   : AIC=-1170.808, Time=0.53 sec
 ARIMA(3,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.58 sec
-> ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=-1173.702, Time=0.47 sec <-
 ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=-1173.097, Time=0.35 sec
 ARIMA(2,1,4)(0,0,0)[0] intercept   : AIC=-1171.594, Time=0.51 sec
 ARIMA(1,1,4)(0,0,0)[0] intercept   : AIC=-1171.150, Time=0.33 sec
 ARIMA(3,1,4)(0,0,0)[0] intercept   : AIC=-1169.378, Time=0.57 sec
 ARIMA(2,1,3)(0,0,0)[0]             : AIC=-1169.545, Time=1.49 sec

Best model:  ARIMA(2,1,3)(0,0,0)[0] intercept
Total fit time: 7.031 seconds
"""

eur_train_AICdiff_fit = auto_arima(y = eur_arima_train['diff'].dropna(),
                                   test="adf",
                                   m=1,
                                   seasonal = False,
                                   stepwise = True,
                                   trace = True
                                   ).fit(y = eur_arima_train['diff'].dropna())

eur_train_AICdiff_fit.summary()

"""
Performing stepwise search to minimize aic
 ARIMA(2,0,2)(0,0,0)[0]             : AIC=-1171.644, Time=0.34 sec
 ARIMA(0,0,0)(0,0,0)[0]             : AIC=-1170.295, Time=0.01 sec
 ARIMA(1,0,0)(0,0,0)[0]             : AIC=-1168.351, Time=0.02 sec
 ARIMA(0,0,1)(0,0,0)[0]             : AIC=-1168.347, Time=0.03 sec
 ARIMA(1,0,2)(0,0,0)[0]             : AIC=-1168.544, Time=0.26 sec
 ARIMA(2,0,1)(0,0,0)[0]             : AIC=-1168.373, Time=0.11 sec
 ARIMA(3,0,2)(0,0,0)[0]             : AIC=-1167.790, Time=0.32 sec
 ARIMA(2,0,3)(0,0,0)[0]             : AIC=-1169.473, Time=0.37 sec
 ARIMA(1,0,1)(0,0,0)[0]             : AIC=-1166.349, Time=0.02 sec
 ARIMA(1,0,3)(0,0,0)[0]             : AIC=-1171.081, Time=0.30 sec
 ARIMA(3,0,1)(0,0,0)[0]             : AIC=-1169.797, Time=0.29 sec
 ARIMA(3,0,3)(0,0,0)[0]             : AIC=-1170.791, Time=0.63 sec
 ARIMA(2,0,2)(0,0,0)[0] intercept   : AIC=-1172.018, Time=3.11 sec
 ARIMA(1,0,2)(0,0,0)[0] intercept   : AIC=-1169.094, Time=0.39 sec
 ARIMA(2,0,1)(0,0,0)[0] intercept   : AIC=-1169.003, Time=0.34 sec
 ARIMA(3,0,2)(0,0,0)[0] intercept   : AIC=-1173.074, Time=0.46 sec
 ARIMA(3,0,1)(0,0,0)[0] intercept   : AIC=-1171.280, Time=0.42 sec
 ARIMA(4,0,2)(0,0,0)[0] intercept   : AIC=-1167.264, Time=0.47 sec
 ARIMA(3,0,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.54 sec
-> ARIMA(2,0,3)(0,0,0)[0] intercept   : AIC=-1173.688, Time=0.47 sec <-
 ARIMA(1,0,3)(0,0,0)[0] intercept   : AIC=-1173.134, Time=0.41 sec
 ARIMA(2,0,4)(0,0,0)[0] intercept   : AIC=-1170.865, Time=0.44 sec
 ARIMA(1,0,4)(0,0,0)[0] intercept   : AIC=-1171.151, Time=0.48 sec
 ARIMA(3,0,4)(0,0,0)[0] intercept   : AIC=-1169.362, Time=0.58 sec

Best model:  ARIMA(2,0,3)(0,0,0)[0] intercept
Total fit time: 10.824 seconds
"""

eur_train_BIC_fit = auto_arima(y = eur_arima_train['eur'],
                               information_criterion="bic",
                               test="adf",
                               m=1,
                               seasonal = False,
                               stepwise = True,
                               trace = True
                               ).fit(y = eur_arima_train['eur'])

eur_train_BIC_fit.summary()

"""
Performing stepwise search to minimize bic
 ARIMA(2,1,2)(0,0,0)[0] intercept   : BIC=-1149.282, Time=0.46 sec
 ARIMA(0,1,0)(0,0,0)[0] intercept   : BIC=-1163.626, Time=0.02 sec
 ARIMA(1,1,0)(0,0,0)[0] intercept   : BIC=-1158.090, Time=0.02 sec
 ARIMA(0,1,1)(0,0,0)[0] intercept   : BIC=-1158.080, Time=0.07 sec
-> ARIMA(0,1,0)(0,0,0)[0]             : BIC=-1166.601, Time=0.01 sec <-
 ARIMA(1,1,1)(0,0,0)[0] intercept   : BIC=-1152.391, Time=0.04 sec

Best model:  ARIMA(0,1,0)(0,0,0)[0]          
Total fit time: 0.628 seconds
"""

eur_train_BIC_fit = auto_arima(y = eur_arima_train['diff'].dropna(),
                               information_criterion="bic",
                               test="adf",
                               m=1,
                               seasonal = False,
                               stepwise = True,
                               trace = True
                               ).fit(y = eur_arima_train['diff'].dropna())

eur_train_BIC_fit.summary()

"""
Performing stepwise search to minimize bic
 ARIMA(2,0,2)(0,0,0)[0]             : BIC=-1153.175, Time=0.36 sec
-> ARIMA(0,0,0)(0,0,0)[0]             : BIC=-1166.601, Time=0.01 sec <-
 ARIMA(1,0,0)(0,0,0)[0]             : BIC=-1160.964, Time=0.02 sec
 ARIMA(0,0,1)(0,0,0)[0]             : BIC=-1160.960, Time=0.06 sec
 ARIMA(1,0,1)(0,0,0)[0]             : BIC=-1155.268, Time=0.03 sec
 ARIMA(0,0,0)(0,0,0)[0] intercept   : BIC=-1163.626, Time=0.02 sec

Best model:  ARIMA(0,0,0)(0,0,0)[0]          
Total fit time: 0.499 seconds
"""

eur_train_AIClag_fit = auto_arima(y = df_eur_exog1,
                                   test="adf",
                                   m=1,
                                   seasonal = False,
                                   stepwise = True,
                                   trace = True
                                   ).fit(y = df_eur_exog1, exog=eur_exog1)

eur_train_AIClag_fit.summary()

"""
Performing stepwise search to minimize aic
 ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=-1127.508, Time=0.40 sec
-> ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=-1131.597, Time=0.02 sec <-
 ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=-1129.752, Time=0.01 sec
 ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=-1129.744, Time=0.04 sec
 ARIMA(0,1,0)(0,0,0)[0]             : AIC=-1131.582, Time=0.01 sec
 ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=-1127.748, Time=0.07 sec

Best model:  ARIMA(0,1,0)(0,0,0)[0] intercept
Total fit time: 0.559 seconds
"""

eur_train_AICdifflag_fit = auto_arima(y = df_eur_exog2,
                                   test="adf",
                                   m=1,
                                   seasonal = False,
                                   stepwise = True,
                                   trace = True
                                   ).fit(y = df_eur_exog2, exog=eur_exog2)

eur_train_AICdifflag_fit.summary()

"""
Performing stepwise search to minimize aic
 ARIMA(2,0,2)(0,0,0)[0]             : AIC=-1128.227, Time=0.29 sec
 ARIMA(0,0,0)(0,0,0)[0]             : AIC=-1131.582, Time=0.01 sec
 ARIMA(1,0,0)(0,0,0)[0]             : AIC=-1129.658, Time=0.02 sec
 ARIMA(0,0,1)(0,0,0)[0]             : AIC=-1129.653, Time=0.05 sec
 ARIMA(1,0,1)(0,0,0)[0]             : AIC=-1127.655, Time=0.02 sec
-> ARIMA(0,0,0)(0,0,0)[0] intercept   : AIC=-1131.597, Time=0.02 sec <-
 ARIMA(1,0,0)(0,0,0)[0] intercept   : AIC=-1129.752, Time=0.01 sec
 ARIMA(0,0,1)(0,0,0)[0] intercept   : AIC=-1129.744, Time=0.06 sec
 ARIMA(1,0,1)(0,0,0)[0] intercept   : AIC=-1127.748, Time=0.04 sec

Best model:  ARIMA(0,0,0)(0,0,0)[0] intercept
Total fit time: 0.532 seconds
"""

eur_train_BIClag_fit = auto_arima(y = df_eur_exog1,
                                  information_criterion="bic",
                                  test="adf",
                                  m=1,
                                  seasonal = False,
                                  stepwise = True,
                                  trace = True
                                  ).fit(y = df_eur_exog1, exog=eur_exog1)

eur_train_BIClag_fit.summary()

"""
Performing stepwise search to minimize bic
 ARIMA(2,1,2)(0,0,0)[0] intercept   : BIC=-1105.551, Time=0.40 sec
-> ARIMA(0,1,0)(0,0,0)[0] intercept   : BIC=-1124.278, Time=0.02 sec <-
 ARIMA(1,1,0)(0,0,0)[0] intercept   : BIC=-1118.774, Time=0.01 sec
 ARIMA(0,1,1)(0,0,0)[0] intercept   : BIC=-1118.766, Time=0.05 sec
 ARIMA(0,1,0)(0,0,0)[0]             : BIC=-1127.922, Time=0.01 sec
 ARIMA(1,1,1)(0,0,0)[0] intercept   : BIC=-1113.110, Time=0.05 sec

Best model:  ARIMA(0,1,0)(0,0,0)[0]          
Total fit time: 0.546 seconds
"""

eur_train_AICdifflag_fit = auto_arima(y = df_eur_exog2,
                                      information_criterion="bic",
                                      test="adf",
                                      m=1,
                                      seasonal = False,
                                      stepwise = True,
                                      trace = True
                                      ).fit(y = df_eur_exog2, exog=eur_exog2)

eur_train_AICdifflag_fit.summary()

"""
Performing stepwise search to minimize bic
 ARIMA(2,0,2)(0,0,0)[0]             : BIC=-1109.930, Time=0.31 sec
-> ARIMA(0,0,0)(0,0,0)[0]             : BIC=-1127.922, Time=0.01 sec <-
 ARIMA(1,0,0)(0,0,0)[0]             : BIC=-1122.339, Time=0.02 sec
 ARIMA(0,0,1)(0,0,0)[0]             : BIC=-1122.334, Time=0.05 sec
 ARIMA(1,0,1)(0,0,0)[0]             : BIC=-1116.677, Time=0.02 sec
 ARIMA(0,0,0)(0,0,0)[0] intercept   : BIC=-1124.278, Time=0.02 sec

Best model:  ARIMA(0,0,0)(0,0,0)[0] 
"""

# In[]: Results from the automated arima:

"""
Basing on AIC

|  AIC SARIMAX(2,1,3) + int. | AIC_diff SARIMAX(2,0,3) + int. |
===============================================================
| Log Likelihood     593.851 | Log Likelihood         593.844 |
| AIC              -1173.702 | AIC                  -1173.688 |
| BIC              -1147.846 | BIC                  -1147.832 |
| intercept           0.0014 | intercept               0.0014 |
| sigma2              0.0011 | sigma2                  0.0011 |
===============================================================

Basing on BIC

|     BIC SARIMAX(0,1,0)    |  BIC_diff SARIMAX(0,0,0)   |
==========================================================
| Log Likelihood    586.148 | Log Likelihood     586.148 |
| AIC             -1170.295 | AIC              -1170.295 |
| BIC             -1166.601 | BIC              -1166.601 |
| sigma2             0.0011 | sigma2              0.0011 |
==========================================================

Basing on AIC with exog

|   Exog1 SARIMAX(0,1,0)    | Exog2 SARIMAX(0,0,0) + int |
==========================================================
| Log Likelihood    567.798 | Log Likelihood     567.798 |
| AIC             -1131.597 | AIC              -1131.597 |
| BIC             -1124.278 | BIC              -1124.278 |
| intercept          0.0028 | intercept           0.0028 |
| sigma2             0.0011 | sigma2              0.0011 |
==========================================================

Basing on BIC with exog

| Exog1 SARIMAX(0,1,0) + int |   Exog2 SARIMAX(0,0,0)     |
===========================================================
| Log Likelihood     566.791 | Log Likelihood     566.791 |
| AIC              -1131.582 | AIC              -1131.582 |
| BIC              -1127.922 | BIC              -1127.922 |
| sigma2              0.0011 | sigma2              0.0011 |
===========================================================
"""