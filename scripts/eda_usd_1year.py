# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # #
# Exploratory Data Analysis = DOLLAR - 1 year #
# # # # # # # # # # # # # # # # # # # # # # # #

# In[0.1]: Importação dos pacotes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import kpss

