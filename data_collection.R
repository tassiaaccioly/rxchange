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

normalizedLastYear <- getNormalizedDate(oneYearBeforeDate)
normalizedYesterday <- getNormalizedDate(yesterdaysDate)

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

## 1. Get buy and sell price average and mean for each day

getAverageByDay <- function(dataSet, columnNumber) {

  temporarySum <- 0
  average <- c()
  dataSetRows <- nrow(dataSet)
  print(dataSetRows)
  averageList <- rep(0,dataSetRows)
  print(averageList)

  # taking in account that each day has 5 quotations, we'll run through these
  # values, sum them, get the average and also save them in a temporary list to
  # get the mean value. We will then save these values in new columns on the
  # dataSet called average and mean

  for (i in 1:dataSetRows) {
    if (i %% 5 == 0) {
      print(dataSet[[i,columnNumber]])
      temporarySum <- temporarySum + dataSet[[i,columnNumber]]

      print(temporarySum)

      average <- temporarySum / 5

      print(average)

      averageList[(i-4):i] <- rep(average,5)

      temporarySum <- 0
      average <- 0

    } else {
      print(dataSet[[i,columnNumber]])
      temporarySum <- temporarySum + dataSet[[i,columnNumber]]
    }
  }

  return(averageList)
}

getAverageAndMeanByDay(clean_eur_series, 1)







