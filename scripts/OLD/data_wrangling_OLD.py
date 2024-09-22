# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 16:25:10 2024

@author: tassi
"""

# In[0.1]: Instalação dos pacotes

!pip install pandas
!pip install numpy
!pip install seaborn
!pip install matplotlib

# In[0.2]: Importação dos pacotes 

import pandas as pd # manipulação de dados em formato de dataframe
import numpy as np # operações matemáticas
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
from datetime import datetime
from dateutil import tz

# In[1.0]: Import datasets

df_usd = pd.read_csv("./scripts/datasets/2024_usd_arima.csv", sep=";", thousands=".", decimal=",", float_precision="high", parse_dates=([0]))

df_eur = pd.read_csv("./scripts/datasets/2024_eur_arima.csv", sep=";", thousands=".", decimal=",", float_precision="high", parse_dates=([0]))


# In[1.1]: Dataset fixing and type handling

df_eur = df_eur.loc[:, ~df_eur.columns.str.contains('^Unnamed')]

df_usd = df_usd.loc[:, ~df_usd.columns.str.contains('^Unnamed')]


# In[2]: Euro Infos and save to csv

df_eur.info()
df_eur.describe()

df_eur = df_eur.rename(columns={"dayTime": "dateTime", "wiseEUR": "eur"})

df_eur.to_csv("./scripts/datasets/wrangled/df_eur.csv")


# In[3]: Dollar Infos and save to csv
df_usd.info()
df_usd.describe()

df_usd = df_usd.rename(columns={"dayTime": "dateTime", "wiseUSD": "usd"})

df_usd.to_csv("./scripts/datasets/wrangled/df_usd.csv")

# In[4]: Plot charts for visual assessment

plt.figure(figsize=(15, 10))
sns.scatterplot(x=df_eur["dateTime"], y=df_eur["eur"], color="limegreen", label="EUR (€)")
sns.scatterplot(x=df_usd["dateTime"], y=df_usd["usd"], color="magenta", label="USD ($)")
plt.axhline(y=np.mean(df_eur["eur"]), color="black", linestyle=":", label="Mean") # mean for euro 
plt.axhline(y=np.mean(df_usd["usd"]), color="black", linestyle=":") # mean for dollar
plt.title("Cotação do Euro e Dólar - Série histórica (17/07/2024 - 14/09/2024)", fontsize="18")
plt.yticks(np.arange(round(df_usd["usd"].min(), 1), round(df_eur["eur"].max() + 0.1, 1), 0.1), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %b, %Y')) 
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10)) 
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Cotação", fontsize="18")
plt.legend(fontsize=18, loc="center right")
plt.show()

# In[5.0]: Import json data and save them as pandas DataFrame

utc_tz = tz.gettz("UTC")

# In[5.1]: Import eur 1year:

with open('./datasets/wise_eur_1year.json') as file:
    data = json.load(file)

df_eur_1year = pd.json_normalize(data)

# delete source and target columns
df_eur_1year.drop(['source', 'target'], axis=1, inplace=True)

# rename columns
df_eur_1year.columns = ["eur", "dateTime"]

# transform dateTime into strings to not upset pandas
df_eur_1year['dateTime'] = df_eur_1year['dateTime'].astype(str)

# iterate over dateTime timestamps and turn them into actual dates.
for index, row in df_eur_1year.loc[:,['dateTime']].iterrows():
    df_eur_1year.at[index, 'dateTime'] = datetime.fromtimestamp(pd.to_numeric(row['dateTime']) / 1000, tz=utc_tz)

# transform dateTime back into date type
df_eur_1year['dateTime'] = pd.to_datetime(df_eur_1year['dateTime'])


# In[5.2]: Import eur 5years

with open('./datasets/wise_eur_5year.json') as file:
    data = json.load(file)

df_eur_5year = pd.json_normalize(data)

df_eur_5year.drop(['source', 'target'], axis=1, inplace=True)

# rename columns
df_eur_5year.columns = ["eur", "dateTime"]

# transform dateTime into strings to not upset pandas
df_eur_5year['dateTime'] = df_eur_5year['dateTime'].astype(str)

# iterate over dateTime timestamps and turn them into actual dates.
for index, row in df_eur_5year.loc[:,['dateTime']].iterrows():
    df_eur_5year.at[index, 'dateTime'] = datetime.fromtimestamp(pd.to_numeric(row['dateTime']) / 1000, tz=utc_tz)

# transform dateTime back into date type
df_eur_5year['dateTime'] = pd.to_datetime(df_eur_5year['dateTime'])

# In[5.3]: Import usd 1year:

with open('./datasets/wise_usd_1year.json') as file:
    data = json.load(file)

df_usd_1year = pd.json_normalize(data)

# delete source and target columns
df_usd_1year.drop(['source', 'target'], axis=1, inplace=True)

# rename columns
df_usd_1year.columns = ["usd", "dateTime"]

# transform dateTime into strings to not upset pandas
df_usd_1year['dateTime'] = df_usd_1year['dateTime'].astype(str)

# iterate over dateTime timestamps and turn them into actual dates.
for index, row in df_usd_1year.loc[:,['dateTime']].iterrows():
    df_usd_1year.at[index, 'dateTime'] = datetime.fromtimestamp(pd.to_numeric(row['dateTime']) / 1000, tz=utc_tz)

# transform dateTime back into date type
df_usd_1year['dateTime'] = pd.to_datetime(df_usd_1year['dateTime'])

# In[5.4]: Import usd 5years

with open('./datasets/wise_usd_5year.json') as file:
    data = json.load(file)

df_usd_5year = pd.json_normalize(data)

df_usd_5year.drop(['source', 'target'], axis=1, inplace=True)

# rename columns
df_usd_5year.columns = ["usd", "dateTime"]

# transform dateTime into strings to not upset pandas
df_usd_5year['dateTime'] = df_usd_5year['dateTime'].astype(str)

# iterate over dateTime timestamps and turn them into actual dates.
for index, row in df_usd_5year.loc[:,['dateTime']].iterrows():
    df_usd_5year.at[index, 'dateTime'] = datetime.fromtimestamp(pd.to_numeric(row['dateTime']) / 1000, tz=utc_tz)

# transform dateTime back into date type
df_usd_5year['dateTime'] = pd.to_datetime(df_usd_5year['dateTime'])