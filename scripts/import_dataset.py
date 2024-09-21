# In[0]: Import libs

import pandas as pd 
import json
from datetime import datetime
from dateutil import tz

# In[1.0]: Import json data and save them as pandas DataFrame

utc_tz = tz.gettz("UTC")

# In[1.1]: Import eur 1year:

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


# In[1.2]: Import eur 5years

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

# In[1.3]: Import usd 1year:

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


# In[1.4]: Import usd 5years

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
df_usd_5year['dateTime'] = pd.to_datetime(df_usd_5year['dateTime'])# -*- coding: utf-8 -*-

