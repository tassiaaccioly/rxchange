import pandas as pd
import numpy as np
import plotly.graph_objects as go
import seaborn as sns

xl_historical_series = pd.ExcelFile('./excel/exchanges_historic_series.xlsx')
df_wiseUSD = pd.read_excel(xl_historical_series, '2024_wise_usd')
df_wiseUSD.head()
df_wiseUSD.info()
df_wiseUSD.describe()

df_wiseEUR = pd.read_excel(xl_historical_series, '2024_wise_eur')
df_wiseEUR.head()
df_wiseEUR.info()
df_wiseEUR.describe()


## Plotting

usd_candlestick = go.Figure(data=[go.Candlestick(x=df_wiseUSD['date'],
                            open=df_wiseUSD['open'],
                            high=df_wiseUSD['high'],
                            low=df_wiseUSD['low'],
                            close=df_wiseUSD['close'])])

usd_candlestick.update_layout(
  title='Wise USD - Série Histórica',
  yaxis_title='Cambio - USD/Real',
  shapes = [dict(x0='2024-07-17', x1='2024-09-07', y0=df_wiseUSD['dailyAvg'], y1=df_wiseUSD['dailyAvg'], xref='x', yref='paper', line_width=2
  )]
)

usd_candlestick.write_html('usd_candlestick.html', auto_open=True)



eur_candlestick = go.Figure(data=[go.Candlestick(x=df_wiseEUR['date'],
                            open=df_wiseEUR['open'],
                            high=df_wiseEUR['high'],
                            low=df_wiseEUR['low'],
                            close=df_wiseEUR['close'])])

eur_candlestick.write_html('eur_candlestick.html', auto_open=True)
