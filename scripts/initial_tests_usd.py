# -*- coding: utf-8 -*-

# In[0.1]: Importação dos pacotes

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# In[0.2]: Importação dos dataframes

df_wg_eur = pd.read_csv("./scripts/datasets/wrangled/df_eur.csv", float_precision="high", parse_dates=([1]))

df_wg_usd = pd.read_csv("./scripts/datasets/wrangled/df_usd.csv", float_precision="high", parse_dates=([1]))

df_wg_eur = df_wg_eur.loc[:, ~df_wg_eur.columns.str.contains('^Unnamed')]

df_wg_usd = df_wg_usd.loc[:, ~df_wg_usd.columns.str.contains('^Unnamed')]

df_wg_eur.info()

df_wg_usd.info()

# In[1.0]: Definindo se as séries são estacionárias ou não

# In[1.1]: Teste de Dickey-Fuller Aumentado para df_wg_eur

eur_adf = adfuller(df_wg_eur["eur"], maxlag=None, autolag="AIC")
eur_adf # não é estacionária, p-value = 0,083 

# (-2.645994127964288,
# 0.0838588367217662,
# 0,
# 179,
# {'1%': -3.4674201432469816,
#  '5%': -2.877826051844538,
#  '10%': -2.575452082332012},
# -723.2729790069569)

eur_adf_ols = adfuller(df_wg_eur['eur'], maxlag=None, autolag="AIC", store=True, regresults=True)[-1].resols.summary()
eur_adf_ols

#                            OLS Regression Results                            
#==============================================================================
#Dep. Variable:                      y   R-squared:                       0.038
#Model:                            OLS   Adj. R-squared:                  0.033
#Method:                 Least Squares   F-statistic:                     7.001
#Date:                Sun, 15 Sep 2024   Prob (F-statistic):            0.00888
#Time:                        21:37:59   Log-Likelihood:                 391.66
#No. Observations:                 179   AIC:                            -779.3
#Df Residuals:                     177   BIC:                            -772.9
#Df Model:                           1                                         
#Covariance Type:            nonrobust                                         
#==============================================================================
#                 coef    std err          t      P>|t|      [0.025      0.975]
#------------------------------------------------------------------------------
#x1            -0.0645      0.024     -2.646      0.009      -0.113      -0.016
#const          0.3967      0.150      2.653      0.009       0.102       0.692
#==============================================================================
#Omnibus:                       39.197   Durbin-Watson:                   2.122
#Prob(Omnibus):                  0.000   Jarque-Bera (JB):              117.546
#Skew:                           0.854   Prob(JB):                     2.99e-26
#Kurtosis:                       6.583   Cond. No.                         462.
#==============================================================================

#Notes:
#[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

# In[1.2]: Teste de Dickey-Fuller Aumentado para df_wg_usd

usd_adf = adfuller(df_wg_usd["usd"], maxlag=None, autolag="AIC")
usd_adf # não é estacionária, p-value = 0,083 

# (-2.3340152630375437,
# 0.1612318910518078,
# 0,
# 182,
# {'1%': -3.4668001583460613,
#  '5%': -2.8775552336674317,
#  '10%': -2.5753075498128246},
# -781.4992404633625)

usd_adf_ols = adfuller(df_wg_usd['usd'], maxlag=None, autolag="AIC", store=True, regresults=True)[-1].resols.summary()
usd_adf_ols

#                            OLS Regression Results                            
#==============================================================================
#Dep. Variable:                      y   R-squared:                       0.029
#Model:                            OLS   Adj. R-squared:                  0.024
#Method:                 Least Squares   F-statistic:                     5.448
#Date:                Sun, 15 Sep 2024   Prob (F-statistic):             0.0207
#Time:                        21:42:06   Log-Likelihood:                 418.92
#No. Observations:                 182   AIC:                            -833.8
#Df Residuals:                     180   BIC:                            -827.4
#Df Model:                           1                                         
#Covariance Type:            nonrobust                                         
#==============================================================================
#                 coef    std err          t      P>|t|      [0.025      0.975]
#------------------------------------------------------------------------------
#x1            -0.0519      0.022     -2.334      0.021      -0.096      -0.008
#const          0.2901      0.124      2.338      0.020       0.045       0.535
#==============================================================================
#Omnibus:                       31.535   Durbin-Watson:                   1.968
#Prob(Omnibus):                  0.000   Jarque-Bera (JB):               87.116
#Skew:                           0.690   Prob(JB):                     1.21e-19
#Kurtosis:                       6.096   Cond. No.                         396.
#==============================================================================

#Notes:
#[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.