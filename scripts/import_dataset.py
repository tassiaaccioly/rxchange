# In[0]: Import libs

import pandas as pd 
import numpy as np
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

# add a log of "eur" column
df_eur_1year['logEUR'] = np.log(df_eur_1year['eur'])

#re-order columns
df_eur_1year = df_eur_1year.reindex(["dateTime", "eur", "logEUR"], axis=1)

# save to csv
df_eur_1year.to_csv("./datasets/wrangled/df_eur_1year.csv", index=False)


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

# add a log of "eur" column
df_eur_5year['logEUR'] = np.log(df_eur_5year['eur'])

#re-order columns
df_eur_5year = df_eur_5year.reindex(["dateTime", "eur", "logEUR"], axis=1)

# save to csv
df_eur_5year.to_csv("./datasets/wrangled/df_eur_5year.csv", index=False)

# In[1.4]: Import usd 3 months

with open('./datasets/wise_eur_3months.json') as file:
    data = json.load(file)

df_eur_3months = pd.json_normalize(data)

df_eur_3months.drop(['source', 'target'], axis=1, inplace=True)

# rename columns
df_eur_3months.columns = ["eur", "dateTime"]

# transform dateTime into strings to not upset pandas
df_eur_3months['dateTime'] = df_eur_3months['dateTime'].astype(str)

# iterate over dateTime timestamps and turn them into actual dates.
for index, row in df_eur_3months.loc[:,['dateTime']].iterrows():
    df_eur_3months.at[index, 'dateTime'] = datetime.fromtimestamp(pd.to_numeric(row['dateTime']) / 1000, tz=utc_tz)

# transform dateTime back into date type
df_eur_3months['dateTime'] = pd.to_datetime(df_eur_3months['dateTime'])

# add a log of "usd" column
df_eur_3months['logEUR'] = np.log(df_eur_3months['eur'])

# add a differenced column
df_eur_3months["diff"] = df_eur_3months["logEUR"].diff()

# remove NA columns
df_eur_3months = df_eur_3months.dropna()

#re-order columns
df_eur_3months = df_eur_3months.reindex(["dateTime", "eur", "logEUR", "diff"], axis=1)

# save to csv
df_eur_3months.to_csv("./datasets/arima_ready/eur_arima_3months.csv", index=False)

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

# add a log of "eur" column
df_usd_1year['logUSD'] = np.log(df_usd_1year['usd'])

#re-order columns
df_usd_1year = df_usd_1year.reindex(["dateTime", "usd", "logUSD"], axis=1)

# save to csv
df_usd_1year.to_csv("./datasets/wrangled/df_usd_1year.csv", index=False)

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
df_usd_5year['dateTime'] = pd.to_datetime(df_usd_5year['dateTime'])

# add a log of "usd" column
df_usd_5year['logUSD'] = np.log(df_usd_5year['usd'])

#re-order columns
df_usd_5year = df_usd_5year.reindex(["dateTime", "usd", "logUSD"], axis=1)

# save to csv
df_usd_5year.to_csv("./datasets/wrangled/df_usd_5year.csv", index=False)

# In[1.4]: Import usd 3 months

with open('./datasets/wise_usd_3months.json') as file:
    data = json.load(file)

df_usd_3months = pd.json_normalize(data)

df_usd_3months.drop(['source', 'target'], axis=1, inplace=True)

# rename columns
df_usd_3months.columns = ["usd", "dateTime"]

# transform dateTime into strings to not upset pandas
df_usd_3months['dateTime'] = df_usd_3months['dateTime'].astype(str)

# iterate over dateTime timestamps and turn them into actual dates.
for index, row in df_usd_3months.loc[:,['dateTime']].iterrows():
    df_usd_3months.at[index, 'dateTime'] = datetime.fromtimestamp(pd.to_numeric(row['dateTime']) / 1000, tz=utc_tz)

# transform dateTime back into date type
df_usd_3months['dateTime'] = pd.to_datetime(df_usd_3months['dateTime'])

# add a log of "usd" column
df_usd_3months['logUSD'] = np.log(df_usd_3months['usd'])

# add a differenced column
df_usd_3months["diff"] = df_usd_3months["logUSD"].diff()

# remove NA columns
df_usd_3months = df_usd_3months.dropna()

#re-order columns
df_usd_3months = df_usd_3months.reindex(["dateTime", "usd", "logUSD", "diff"], axis=1)

# save to csv
df_usd_3months.to_csv("./datasets/arima_ready/usd_arima_3months.csv", index=False)
