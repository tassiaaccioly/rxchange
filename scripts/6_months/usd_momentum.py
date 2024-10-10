# -*- coding: utf-8 -*-

# In[0]: Import packages and set variables

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from itertools import repeat

dt_format = "%d/%m/%Y"

# In[0.2]: Import databases

usd_1year = pd.read_csv("./datasets/wrangled/df_usd_1year.csv", float_precision="high", parse_dates=([1]))

usd_6months = usd_1year[185:].drop(["weekday", "date", "diff", "log"], axis=1)

usd_6months = usd_6months.set_index(np.arange(0,180,1))

# 25/03/2024 - 20/09/2024

# In[0.3]: Getting the rolling mean

rollingSum = 0
count = 0
for index, value in enumerate(usd_6months["usd"]):
    count += 1
    if index == 0:
        rollingSum += usd_6months.iloc[index]["usd"]
        usd_6months.at[index, 'rollMean'] = usd_6months.iloc[index]["usd"]
        print(rollingSum, count)
    else:
        rollingSum += usd_6months.iloc[index]["usd"]
        usd_6months.at[index, 'rollMean'] = np.round(rollingSum / count, 5)
        print(rollingSum, count)

# In[1.1]: Reindex the database:

usd_6months = pd.DataFrame(usd_6months.drop("dateTime", axis=1).values, index=pd.date_range(start="2024-03-25", periods=180), columns=["usd", "rollMean"])

# 25/03/2024 - 20/09/2024

# In[1.1]: Getting the short moving average

usd_6months["roll_5days"] = usd_6months["usd"].rolling(window=5).mean()
        

# In [1.2]: Getting the long moving average

usd_6months["roll_15days"] = usd_6months["usd"].rolling(window=15).mean()

# In [1.2]: Getting the long moving average

usd_6months["roll_30days"] = usd_6months["usd"].rolling(window=30).mean()

# In[1.3]: Plotting all the means:

sns.set_style("whitegrid", {"grid.linestyle": ":"})
sns.set_palette("viridis_r", n_colors=7)
fig, ax = plt.subplots(1, figsize=(15, 10), dpi=600)
ax.spines['bottom'].set(linewidth=3, color="black")
ax.spines['left'].set(linewidth=3, color="black")
sns.lineplot(usd_6months["usd"], label="Câmbio USD", linewidth=3)
sns.lineplot(usd_6months["rollMean"], label="Média Acumulada", linewidth=3, linestyle=":")
sns.lineplot(usd_6months["roll_5days"].dropna(), label="Média Móvel (5 dias)", linewidth=3, linestyle="--")
sns.lineplot(usd_6months["roll_15days"].dropna(), label="Média Móvel (15 dias)", linewidth=3)
sns.lineplot(usd_6months["roll_30days"].dropna(), label="Média Móvel (30 dias)", linewidth=3, linestyle="-.")
#sns.lineplot(y=list(repeat(np.max(usd_6months["usd"]), len(usd_6months))), x=usd_6months.index, lw=3, linestyle="--", label="Máxima") # max usd
#sns.lineplot(y=list(repeat(np.mean(usd_6months["usd"]), len(usd_6months))), x=usd_6months.index, lw=3, linestyle="--", label="Média") # mean usd
#sns.lineplot(y=list(repeat(np.min(usd_6months["usd"]), len(usd_6months))), x=usd_6months.index, lw=3, linestyle="--", label="Mínima") # min usd
# plt.title(f'Médias Móveis do Dólar - Série histórica ({usd_6months.index[0].strftime(dt_format)} - {usd_6months.index[-1].strftime(dt_format)})', fontsize="18")
plt.yticks(np.arange(4.95, round(usd_6months["usd"].max(), 1), 0.05), fontsize="22")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=20))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="22")
plt.xlabel("", fontsize="22")
plt.ylabel("Câmbio USD ↔ BRL", fontsize="22")
plt.legend(fontsize="22", loc="upper left")
plt.show()
