# Source helper functions:
source("helper_functions.R")

# Gets the exchange values
# TODO get the coin/exchange name from user input

euroCode <- "EUR"
dollarCode <- "USD"

currentExchange <- "USD"

# Define dates for creating the API url:
# Getting final historical series date:

todaysDate <- now()
# yesterdaysDate <- todaysDate - days(1)

initialDate <- date("2024-04-05")


# Getting initial historical series date:

oneYearBeforeDate <- initialDate - years(1)

# Normalize dates to fit API style

normalizedLastYear <- getNormalizedDate(oneYearBeforeDate, "us")
normalizedYesterday <- getNormalizedDate(todaysDate - days(1), "us")

normalizedYesterday <- "06-28-2024"

# Get historical series from Banco do Brasil API for dollar

# series_name <- stringr::str_interp("${tolower(dollarCode)}_series")

# assign(series_name, getHistoricalSeries(dollarCode, normalizedLastYear, normalizedYesterday))

# Get historical series from Banco do Brasil API for euro

series_name <- stringr::str_interp("${tolower(currentExchange)}_series")

assign(series_name, getHistoricalSeries(currentExchange, normalizedLastYear, normalizedYesterday))

# Cleaning the data sets

## 1. Turn the data_frame into a tibble and remove the "paridade" columns

clean_eur_series <- tibble::as_tibble(eur_series[,c(3,5,6)])
clean_usd_series <- tibble::as_tibble(usd_series[,c(3,5,6)])

### 1a. Rename columns

columnNames <- c("buyPrice", "date", "priceType")

names(clean_eur_series) <- columnNames
names(clean_usd_series) <- columnNames

## 2. Transforming priceType into a factor and adding a new levels columns

typeFactor <- as.factor(clean_eur_series$priceType)

clean_eur_series$typeLevels <- as.numeric(typeFactor)
clean_eur_series$priceType <- typeFactor

typeFactor <- as.factor(clean_usd_series$priceType)

clean_usd_series$typeLevels <- as.numeric(typeFactor)
clean_usd_series$priceType <- typeFactor

## 3. Cleaning date values

clean_eur_series$date <- clean_eur_series$date %>% date()

clean_usd_series$date <- clean_usd_series$date %>% date()

# Generate more data from the data set

## 1. Get buy price mean for each day

clean_eur_series <- clean_eur_series %>% mutate(
  day = cumsum(row_number() %% 5 == 1)) %>% mutate(
    dailyAverageBuy = mean(buyPrice),
    .by = day
  )

clean_usd_series <- clean_usd_series %>% mutate(
  day = cumsum(row_number() %% 5 == 1)) %>% mutate(
    dailyAverageBuy = mean(buyPrice),
    .by = day
  )

# Change columns places

clean_eur_series <- clean_eur_series %>% select(day, date, buyPrice, dailyAverageBuy, priceType, typeLevels)
clean_usd_series <- clean_usd_series %>% select(day, date, buyPrice, dailyAverageBuy, priceType, typeLevels)

## 2. Get general mean for the whole dataset:

### 2a. Buy price

historicalAverageBuyEuro <- mean(clean_eur_series$buyPrice)
historicalAverageBuyDolar <- mean(clean_usd_series$buyPrice)

## 1. Plot the line graph of the database

eur_plot <- ggplot(clean_eur_series) +
  aes(x = date, y = dailyAverageBuy) +
  geom_line(linewidth = 1L, colour = "black") +
  scale_x_date(breaks = scales::breaks_pretty(n = 6)) +
  labs(
    x = "Mês",
    y = "Média diária real/euro (compra)",
    title = "", # "Série histórica de taxa de câmbio",
    subtitle = "", # stringr::str_interp("${currentExchange} (compra)"),
    caption = "" # stringr::str_interp("${normalizedLastYear} - ${normalizedYesterday}")
  ) +
  geom_line(aes(y = mean(buyPrice)), colour = "red", linewidth = 0.8, linetype = "dashed", show.legend = TRUE) +
  theme_bw() +
  theme(
    axis.text.y = element_text(size = 11L, colour = "black"),
    axis.text.x = element_text(size = 11L, colour = "black"),
    axis.line = element_line(colour = "black", size = 1.5),
    text = element_text(size = 13L)
  ) 

usd_plot <- ggplot(clean_usd_series) +
  aes(x = date, y = dailyAverageBuy) +
  geom_line(linewidth = 1L, colour = "black") +
  scale_x_date(breaks = scales::breaks_pretty(n = 6)) +
  labs(
    x = "Mês",
    y = "Média diária real/dólar (compra)",
    title = "", # "Série histórica de taxa de câmbio",
    subtitle = "", # stringr::str_interp("${currentExchange} (compra)"),
    caption = "" # stringr::str_interp("${normalizedLastYear} - ${normalizedYesterday}")
  ) +
  geom_line(aes(y = mean(buyPrice)), colour = "red", linewidth = 0.8, linetype = "dashed", show.legend = TRUE) +
  theme_bw() +
  theme(
    axis.text.y = element_text(size = 11L, colour = "black"),
    axis.text.x = element_text(size = 11L, colour = "black"),
    axis.line = element_line(colour = "black", size = 1.5),
    text = element_text(size = 13L)
  ) 

eur_plot + lims(y = c(4.7, 6))

usd_plot + lims(y = c(4.7, 6))

## 3. Get descriptives for the databases

descritivasEuro <- describe(clean_eur_series$dailyAverageBuy)

descritivasEuro2 <- clean_eur_series %>% summarise(
  standardDeviation = sd(dailyAverageBuy), 
  standartError = sd(dailyAverageBuy)/sqrt(length(dailyAverageBuy)),
  variance = var(dailyAverageBuy), 
  moda = getMode(dailyAverageBuy),
  mean = mean(dailyAverageBuy),
  máximo = max(dailyAverageBuy),
  mínimo = min(dailyAverageBuy),
  elementsUnderMean = length(which(dailyAverageBuy < mean(dailyAverageBuy)))
  )

length(which(clean_eur_series$dailyAverageBuy < 5.402)) #valores abaixo da média
length(which(clean_eur_series$dailyAverageBuy < 5.362)) #valores abaixo da mediana
length(which(clean_eur_series$dailyAverageBuy < 5.456)) #valores abaixo do terceiro quartil

descritivasDolar <- describe(clean_usd_series$dailyAverageBuy)

descritivasDolar2 <- clean_usd_series %>% summarise(
  standardDeviation = sd(dailyAverageBuy), 
  standartError = sd(dailyAverageBuy)/sqrt(length(dailyAverageBuy)),
  variance = var(dailyAverageBuy), 
  moda = getMode(dailyAverageBuy),
  mean = mean(dailyAverageBuy),
  máximo = max(dailyAverageBuy),
  mínimo = min(dailyAverageBuy),
  elementsUnderMean = length(which(dailyAverageBuy < mean(dailyAverageBuy)))
)

length(which(clean_usd_series$dailyAverageBuy < 4.989)) #valores abaixo da média
length(which(clean_usd_series$dailyAverageBuy < 4.960)) #valores abaixo da mediana
length(which(clean_usd_series$dailyAverageBuy < 5.053)) #valores abaixo do terceiro quartil

# Calculating rolling means (5-5 days/Weekly)

## For Euro

### Select only the closing values
better_eur_statistics <- clean_eur_series[clean_eur_series$priceType == "Fechamento",]

### calculate rolling mean of the closing values
eurRollMeanAvg <- rollmean(better_eur_statistics$dailyAverageBuy, 5, fill=NA, align="right")

eurRollMean <- rollmean(better_eur_statistics$buyPrice, 5, fill=NA, align="right")

### adds the moving averages back into the new dataset
better_eur_statistics <- better_eur_statistics %>% mutate(rollMean = eurRollMean, rollMeanAvg = eurRollMeanAvg)

## For Dollar

better_usd_statistics <- clean_usd_series[clean_usd_series$priceType == "Fechamento",]

usdRollMeanAvg <- rollmean(better_usd_statistics$dailyAverageBuy, 5, fill=NA, align="right")

usdRollMean <- rollmean(better_usd_statistics$buyPrice, 5, fill=NA, align="right")

better_usd_statistics <- better_usd_statistics %>% mutate(rollMean = usdRollMean, rollMeanAvg = usdRollMeanAvg)

# Calculating the risk

# Fixing the eur_series dataframe

test_eur_series <- eur_series[,c(3,5,6)]

names(test_eur_series) <- columnNames

test_eur_series$date <- test_eur_series$date %>% date() 

# make a separate date tibble to add proper id
eur_series_dates <- test_eur_series$date

eur_series_dates <- eur_series_dates[!duplicated(eur_series_dates)]

eur_series_dates <- tibble(eur_series_dates) %>% mutate(id = row_number()) %>% rename(date = eur_series_dates)

## get only the opening rows
test_eur_series_trans <- test_eur_series[test_eur_series$priceType == "Abertura",]

## rename column and remove priceType Column
test_eur_series_trans <- test_eur_series_trans %>% rename(openingPrice = buyPrice) %>% mutate(priceType = NULL)

## left join ids to database
test_eur_series_trans <- left_join(test_eur_series_trans, eur_series_dates, by = "date")

## get only the closing rows, rename price column and delete other columns
test_eur_closing_rows <- test_eur_series[test_eur_series$priceType == 'Fechamento',] %>% rename(closingPrice = buyPrice) %>% mutate(priceType = NULL)

## join closing prices in test_eur_series_trans
test_eur_series_trans <- left_join(test_eur_series_trans, test_eur_closing_rows, by = "date")

## reorganizing the columns
test_eur_series_trans <- test_eur_series_trans %>% select(id, date, openingPrice, closingPrice)

## Organizing the intermediate values:

### get only the intermediate values
test_eur_series_inter <- test_eur_series[test_eur_series$priceType == "Intermediário",]

### adds ids to the database
test_eur_series_inter <- left_join(test_eur_series_inter, eur_series_dates, by = "date")

### get intermediary value average for each day, remove unnecessary columns and reorganize
test_eur_series_inter <- test_eur_series_inter %>% mutate(intermediaryPrice = mean(buyPrice), .by = "id") %>% mutate(buyPrice = NULL, priceType = NULL, id = NULL) %>% select(date, intermediaryPrice)

### remove duplicate rows
test_eur_series_inter <- test_eur_series_inter[!duplicated(test_eur_series_inter),]

## join intermediary prices in test_eur_series_trans
test_eur_series_trans <- left_join(test_eur_series_trans, test_eur_series_inter, by = "date")



