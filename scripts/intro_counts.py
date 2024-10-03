# In[0.1]: Importação dos pacotes

import numpy as np
import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.api import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import shapiro, jarque_bera
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# In[0.2]: Import dataframes

df_usd_arima_5years = pd.read_csv("./datasets/wrangled/df_usd_5year.csv", float_precision="high", parse_dates=([1]))

df_usd_arima_5years.info()

usd_intro = df_usd_arima_5years.drop(["log", "diff", "date"], axis=1)

usd_intro = usd_intro.drop(usd_intro[usd_intro["dateTime"].dt.strftime("%Y") != "2023"].index)

usd_intro = usd_intro.reset_index().drop(["index"], axis=1)

# Create a rolling mean column in usd_intro:

for index, row in usd_intro.loc[:,['usd']].iterrows():
    usd_len = len(usd_intro)
    if index == 0:
        usd_intro.at[index, "rollingMean"] = row["usd"]
    elif index == 364:
        usd_intro.at[index, "rollingMean"] = np.mean(usd_intro["usd"])
    else:
        usd_intro.at[index, "rollingMean"] = np.mean(usd_intro["usd"][:-(usd_len - 1 - index)])

dt_format = "%d/%m/%Y"

# In[1.0]: Generate Statistics for this dataset

# Plot the dataset with average, max and min:

plt.style.use("seaborn-v0_8-colorblind")
sns.set_style("whitegrid",  {"grid.linestyle": ":"})
plt.figure(figsize=(15, 10), dpi=300)
sns.lineplot(y = usd_intro["usd"], x = usd_intro["dateTime"], color="limegreen", label="Câmbio usd")
sns.lineplot(y = usd_intro["rollingMean"], x = usd_intro["dateTime"], color="darkorchid", label="Média Móvel")
plt.axhline(y=np.mean(usd_intro["usd"][:-180]), color="black", linestyle="--", label="Média até Julho", linewidth=2) # mean for usd
plt.axhline(y=np.max(usd_intro["usd"]), color="magenta", label="Máxima", linewidth=2) # max for usd
plt.axhline(y=np.min(usd_intro["usd"]), color="magenta", linestyle="--", label="Mínima", linewidth=2) # min for usd
plt.title("Cotação do Dólar - Série histórica 2023", fontsize="18")
plt.yticks(np.arange(round(usd_intro["usd"].min(), 1), round(usd_intro["usd"].max() + 0.1, 1), 0.1), fontsize="14")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d, %b'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))
plt.gca().xaxis.set_tick_params(rotation = -30)
plt.xticks(fontsize="14")
plt.xlabel("Data", fontsize="18")
plt.ylabel("Câmbio USD ↔ BRL", fontsize="18")
plt.legend(fontsize=18, loc="upper right", bbox_to_anchor=(0, 0.93, 0.98, 0))
plt.show()

runningMean = 0
count = 0
lowest = 4.79
day = 0

for index, value in enumerate(usd_intro["usd"]):
    firstMean = 5.06
    if (index > 180 and value <= usd_intro["rollingMean"][index]):
        count += 1
        runningMean += usd_intro["rollingMean"][index] - value
        
        if value < usd_intro["rollingMean"][index]:
            lowest = value
            day = index
            print(f"{count}: {day} - {lowest}")

runningMean / count




# np.mean(usd_intro["usd"][:-180]) // média no primeiro semestre
# 5.06708027027027

# np.max(usd_intro["usd"][-180:]) // máxima no segundo semestre
# 5.1677

# np.min(usd_intro["usd"][-180:]) // valor mínimo no segundo semestre
# 4.7246

# Média de economia segundo semestre:
# 0.14862

# Média de economia segundo semestre abaixo da média de Jan-Jun:
# 0.1608

# Números de dias com valores abaixo de 4,79
# 13

# Média de economia dos dias com valores abaixo de 4,79
# 0,03