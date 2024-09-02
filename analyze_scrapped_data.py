import pandas as pd
import numpy as np

xl_historical_series = pd.ExcelFile('./excel/exchanges_historic_series.xls')
df_wiseUSD = pd.read_excel(xl_historical_series, '2024_wise_usd')
df_wiseUSD.head()

df_wiseEUR = pd.read_excel(xl_historical_series, '2024_wise_eur')
df_wiseEUR.head()
df_wiseEUR.describe()
