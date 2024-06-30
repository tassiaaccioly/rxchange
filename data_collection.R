# Source helper functions:
source("helper_functions.R")

# Define dates for creating the API url:

euroCode <- "EUR"
dollarCode <- "USD"

# Getting final historical series date:

todaysDate <- now()
yesterdaysDate <- todaysDate - days(1)

# Getting initial historical series date:

oneYearBeforeDate <- todaysDate - years(1)

# Normalize dates to fit API style

normalizedLastYear <- getNormalizedDate(oneYearBeforeDate, "us")
normalizedYesterday <- getNormalizedDate(yesterdaysDate, "us")

# Get historical series from Banco do Brasil API for dollar

# series_name <- stringr::str_interp("${tolower(dollarCode)}_series")

# assign(series_name, getHistoricalSeries(dollarCode, normalizedLastYear, normalizedYesterday))

# Get historical series from Banco do Brasil API for euro

series_name <- stringr::str_interp("${tolower(euroCode)}_series")

assign(series_name, getHistoricalSeries(euroCode, normalizedLastYear, normalizedYesterday))

# Cleaning the data sets

## 1. Turn the data_frame into a tibble and remove the "paridade" columns

clean_eur_series <- tibble::as_tibble(eur_series[,3:6])

### 1a. Rename columns

columnNames <- c("buyPrice", "sellPrice", "date", "priceType")

names(clean_eur_series) <- columnNames

## 2. Transforming priceType into a factor and adding a new levels columns

typeFactor <- as.factor(clean_eur_series$priceType)

clean_eur_series$typeLevels <- as.numeric(typeFactor)
clean_eur_series$priceType <- typeFactor

## 3. Cleaning date values

clean_eur_series$date <- clean_eur_series$date %>% date()

# Generate more data from the data set

## 1. Get buy and sell price mean for each day

clean_eur_series <- clean_eur_series %>% mutate(
  group = cumsum(row_number() %% 5 == 1)) %>% mutate(
    dailyAverageBuy = mean(buyPrice),
    dailyAverageSell = mean(sellPrice),
    .by = group
  )

## 2. Get general mean for the whole dataset:

### 2a. Buy price

historicalAverageBuy <- mean(clean_eur_series$buyPrice)

historicalAverageSell <- mean(clean_eur_series$sellPrice)

### 2b. Sell price

## 1. Plot the line graph of the database

# buyPrice:

ggplot(clean_eur_series) +
  aes(x = date, y = dailyAverageBuy) +
  geom_line(linewidth = 1L, colour = "#0D3053") +
  labs(
    x = "",
    y = "Média diária real/euro (compra)",
    title = "Série histórica de taxa de câmbio",
    subtitle = stringr::str_interp("${currentExchange} (compra)"),
    caption = stringr::str_interp("${normalizedLastYear} - ${normalizedYesterday}")
  ) +
  geom_line(aes(y = mean(buyPrice)), colour = "red", linewidth = 0.8, linetype = "dashed") +
  theme_bw() +
  theme(
    axis.text.y = element_text(size = 11L),
    axis.text.x = element_text(size = 11L)
  )

# sellPrice

ggplot(clean_eur_series) +
  aes(x = date, y = dailyAverageSell) +
  geom_line(linewidth = 1L, colour = "#0D3053") +
  labs(
    x = "",
    y = "Média diária real/euro (venda)",
    title = "Série histórica de taxa de câmbio",
    subtitle = stringr::str_interp("${currentExchange} (venda)"),
    caption = stringr::str_interp("${normalizedLastYear} - ${normalizedYesterday}")
  ) +
  geom_line(aes(y = mean(sellPrice)), colour = "red", linewidth = 0.8, linetype = "dashed") +
  theme_bw() +
  theme(
    axis.text.y = element_text(size = 11L),
    axis.text.x = element_text(size = 11L)
  )