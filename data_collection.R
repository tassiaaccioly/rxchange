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

clean_eur_series$date <- clean_eur_series$date %>% as.Date(clean_eur_series$date)

# Generate more data from the data set

clean_eur_series["buyPrice"][1]

## 1. Get buy and sell price average and mean for each day

clean_eur_series <- clean_eur_series %>% mutate(
  group = cumsum(row_number() %% 5 == 1)) %>% mutate(
    dailyAverageBuy = mean(buyPrice),
    dailyAverageSell = mean(sellPrice),
    .by = group
    )


