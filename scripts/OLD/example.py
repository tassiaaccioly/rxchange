# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 17:26:04 2024

@author: tassi
"""

import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.stattools import grangercausalitytests
from scipy import stats

# Exemplo de dados de séries temporais
dados = df_wg_eur_1year

# Se quiser aplicar a transformação logarítmica
# dados_log = np.log(dados[""])

# Ajuste automático do ARIMA
modelo_ajustado = pm.auto_arima(dados['logEUR'],
seasonal=True, # Para SARIMA, defina como True
stepwise=True,
suppress_warnings=True,
trace=True)
# Exibe o melhor modelo selecionado
modelo_ajustado.summary()
# Fazendo previsões
previsao = modelo_ajustado.predict(n_periods=10)
print(previsao)
# Teste de Shapiro-Wilk para normalidade dos resíduos
residuos = modelo_ajustado.resid()
residuos
shapiro_test = stats.shapiro(residuos)
print(f'Shapiro-Wilk test p-value: {shapiro_test.pvalue}')

# Teste de causalidade de Granger (lag de até 4 períodos)
# Exemplo para verificar se a série "Y" causa "X" no sentido de Granger
dados_granger = pd.concat([dados['logEUR'], dados['logEUR']], axis=1)
resultado_granger = grangercausalitytests(dados_granger, maxlag=4)
