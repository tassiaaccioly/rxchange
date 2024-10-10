# In[0]: Import libs

import pandas as pd 
import numpy as np
import json
from datetime import datetime
from dateutil import tz

# In[1.0]: Import json data and save them as pandas DataFrame

utc_tz = tz.gettz("UTC")

# In[1.1]: Import eur 1year:

with open('../datasets/wise_eur_1year.json') as file:
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

# Add a column of the weekday (so we can drop sundays as they are skewing the data)
for index, row in df_eur_1year.loc[:,['dateTime']].iterrows():
    df_eur_1year.at[index, 'weekday'] = row['dateTime'].strftime("%A")
    df_eur_1year.at[index, 'date'] = row['dateTime'].strftime('%Y-%m-%d')

# Remove all "Sundays" from the dataframe
#df_eur_1year = df_eur_1year.drop(df_eur_1year[df_eur_1year["weekday"] == "Sunday"].index)

# add a log of "eur" column
df_eur_1year['log'] = np.log(df_eur_1year['eur'])

# add a "diff" column
df_eur_1year["diff"] = df_eur_1year["eur"].diff()

#re-order columns
df_eur_1year = df_eur_1year.reindex(["date", "dateTime", "eur", "log", "diff", "weekday"], axis=1)

# save to csv
df_eur_1year.to_csv("../datasets/wrangled/df_eur_1year.csv", index=False)


# In[1.2]: Import eur 5years

with open('../datasets/wise_eur_5year.json') as file:
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

# Add a column of the weekday (so we can drop sundays as they are skewing the data)
for index, row in df_eur_5year.loc[:,['dateTime']].iterrows():
    df_eur_5year.at[index, 'weekday'] = row['dateTime'].strftime("%A")
    df_eur_5year.at[index, 'date'] = row['dateTime'].strftime('%Y-%m-%d')

# Remove all "Sundays" from the dataframe
#df_eur_5year = df_eur_5year.drop(df_eur_5year[df_eur_5year["weekday"] == "Sunday"].index)

# add a log of "eur" column
df_eur_5year['log'] = np.log(df_eur_5year['eur'])

# add a "diff" column
df_eur_5year["diff"] = df_eur_5year["eur"].diff()

#re-order columns
df_eur_5year = df_eur_5year.reindex(["date", "dateTime", "eur", "log", "diff", "weekday"], axis=1)

# save to csv
df_eur_5year.to_csv("../datasets/wrangled/df_eur_5year.csv", index=False)

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

# Add a column of the weekday (so we can drop sundays as they are skewing the data)
for index, row in df_eur_3months.loc[:,['dateTime']].iterrows():
    df_eur_3months.at[index, 'weekday'] = row['dateTime'].strftime("%A")
    df_eur_3months.at[index, 'date'] = row['dateTime'].strftime('%Y-%m-%d')

# Remove all "Sundays" from the dataframe
#df_eur_3months = df_eur_3months.drop(df_eur_3months[df_eur_3months["weekday"] == "Sunday"].index)

# add a log of "eur" column
df_eur_3months['log'] = np.log(df_eur_3months['eur'])

# add a "diff" column
df_eur_3months["diff"] = df_eur_3months["eur"].diff()

#re-order columns
df_eur_3months = df_eur_3months.reindex(["date", "dateTime", "eur", "log", "diff", "weekday"], axis=1)

# save to csv
df_eur_3months.to_csv("./datasets/wrangled/df_eur_3months.csv", index=False)

# In[1.4]: Import usd 10 days

with open('./datasets/wise_eur_10days.json') as file:
    data = json.load(file)

df_eur_10days = pd.json_normalize(data)

df_eur_10days.drop(['source', 'target'], axis=1, inplace=True)

# rename columns
df_eur_10days.columns = ["eur", "dateTime"]

# transform dateTime into strings to not upset pandas
df_eur_10days['dateTime'] = df_eur_10days['dateTime'].astype(str)

# iterate over dateTime timestamps and turn them into actual dates.
for index, row in df_eur_10days.loc[:,['dateTime']].iterrows():
    df_eur_10days.at[index, 'dateTime'] = datetime.fromtimestamp(pd.to_numeric(row['dateTime']) / 1000, tz=utc_tz)

# transform dateTime back into date type
df_eur_10days['dateTime'] = pd.to_datetime(df_eur_10days['dateTime'])

# Add a column of the weekday (so we can drop sundays as they are skewing the data)
for index, row in df_eur_10days.loc[:,['dateTime']].iterrows():
    df_eur_10days.at[index, 'weekday'] = row['dateTime'].strftime("%A")
    df_eur_10days.at[index, 'date'] = row['dateTime'].strftime('%Y-%m-%d')

# Remove all "Sundays" from the dataframe
#df_usd_3months = df_eur_3months.drop(df_usd_3months[df_usd_3months["weekday"] == "Sunday"].index)

# add a log of "eur" column
df_eur_10days['log'] = np.log(df_eur_10days['eur'])

# add a "diff" column
df_eur_10days["diff"] = df_eur_10days["eur"].diff()

#re-order columns
df_eur_10days = df_eur_10days.reindex(["date", "dateTime", "eur", "log", "diff", "weekday"], axis=1)

# save to csv
df_eur_10days.to_csv("./datasets/wrangled/df_eur_10days.csv", index=False)

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

# Add a column of the weekday (so we can drop sundays as they are skewing the data)
for index, row in df_usd_1year.loc[:,['dateTime']].iterrows():
    df_usd_1year.at[index, 'weekday'] = row['dateTime'].strftime("%A")
    df_usd_1year.at[index, 'date'] = row['dateTime'].strftime('%Y-%m-%d')

# Remove all "Sundays" from the dataframe
#df_usd_1year = df_usd_1year.drop(df_usd_1year[df_usd_1year["weekday"] == "Sunday"].index)

# add a log of "eur" column
df_usd_1year['log'] = np.log(df_usd_1year['usd'])

# add a "diff" column
df_usd_1year["diff"] = df_usd_1year["usd"].diff()

#re-order columns
df_usd_1year = df_usd_1year.reindex(["date", "dateTime", "usd", "log", "diff", "weekday"], axis=1)

# save to csv
df_usd_1year.to_csv("./datasets/wrangled/df_usd_1year.csv", index=False)

# In[1.4]: Import usd 5years

with open('../datasets/wise_usd_5year.json') as file:
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

# Add a column of the weekday (so we can drop sundays as they are skewing the data)
for index, row in df_usd_5year.loc[:,['dateTime']].iterrows():
    df_usd_5year.at[index, 'weekday'] = row['dateTime'].strftime("%A")
    df_usd_5year.at[index, 'date'] = row['dateTime'].strftime('%Y-%m-%d')

# Remove all "Sundays" from the dataframe
#df_usd_5year = df_usd_5year.drop(df_usd_5year[df_usd_5year["weekday"] == "Sunday"].index)

# add a log of "eur" column
df_usd_5year['log'] = np.log(df_usd_5year['usd'])

# add a "diff" column
df_usd_5year["diff"] = df_usd_5year["usd"].diff()

#re-order columns
df_usd_5year = df_usd_5year.reindex(["date", "dateTime", "usd", "log", "diff", "weekday"], axis=1)

# save to csv
df_usd_5year.to_csv("../datasets/wrangled/df_usd_5year.csv", index=False)

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

# Add a column of the weekday (so we can drop sundays as they are skewing the data)
for index, row in df_usd_3months.loc[:,['dateTime']].iterrows():
    df_usd_3months.at[index, 'weekday'] = row['dateTime'].strftime("%A")
    df_usd_3months.at[index, 'date'] = row['dateTime'].strftime('%Y-%m-%d')

# Remove all "Sundays" from the dataframe
#df_usd_3months = df_eur_3months.drop(df_usd_3months[df_usd_3months["weekday"] == "Sunday"].index)

# add a log of "eur" column
df_usd_3months['log'] = np.log(df_usd_3months['usd'])

# add a "diff" column
df_usd_3months["diff"] = df_usd_3months["usd"].diff()

#re-order columns
df_usd_3months = df_usd_3months.reindex(["date", "dateTime", "usd", "log", "diff", "weekday"], axis=1)

# save to csv
df_usd_3months.to_csv("./datasets/wrangled/df_usd_3months.csv", index=False)

# In[1.4]: Import usd 10 days

with open('./datasets/wise_usd_10days.json') as file:
    data = json.load(file)

df_usd_10days = pd.json_normalize(data)

df_usd_10days.drop(['source', 'target'], axis=1, inplace=True)

# rename columns
df_usd_10days.columns = ["usd", "dateTime"]

# transform dateTime into strings to not upset pandas
df_usd_10days['dateTime'] = df_usd_10days['dateTime'].astype(str)

# iterate over dateTime timestamps and turn them into actual dates.
for index, row in df_usd_10days.loc[:,['dateTime']].iterrows():
    df_usd_10days.at[index, 'dateTime'] = datetime.fromtimestamp(pd.to_numeric(row['dateTime']) / 1000, tz=utc_tz)

# transform dateTime back into date type
df_usd_10days['dateTime'] = pd.to_datetime(df_usd_10days['dateTime'])

# Add a column of the weekday (so we can drop sundays as they are skewing the data)
for index, row in df_usd_10days.loc[:,['dateTime']].iterrows():
    df_usd_10days.at[index, 'weekday'] = row['dateTime'].strftime("%A")
    df_usd_10days.at[index, 'date'] = row['dateTime'].strftime('%Y-%m-%d')

# Remove all "Sundays" from the dataframe
#df_usd_3months = df_eur_3months.drop(df_usd_3months[df_usd_3months["weekday"] == "Sunday"].index)

# add a log of "eur" column
df_usd_10days['log'] = np.log(df_usd_10days['usd'])

# add a "diff" column
df_usd_10days["diff"] = df_usd_10days["usd"].diff()

#re-order columns
df_usd_10days = df_usd_10days.reindex(["date", "dateTime", "usd", "log", "diff", "weekday"], axis=1)

# save to csv
df_usd_10days.to_csv("./datasets/wrangled/df_usd_10days.csv", index=False)
