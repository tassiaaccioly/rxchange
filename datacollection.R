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

normalizedLastYear <- getReversedDate(oneYearBeforeDate)
normalizedYesterday <- getReversedDate(yesterdaysDate)

# Get infos from Banco do Brasil API for dollar

series_name <- stringr::str_interp("${tolower(dollarCode)}_series")

assign(series_name, getHistoricalSeries(dollarCode, normalizedLastYear, normalizedYesterday))

# Get infos from Banco do Brasil API for euro

series_name <- stringr::str_interp("${tolower(euroCode)}_series")

assign(series_name, getHistoricalSeries(euroCode, normalizedLastYear, normalizedYesterday))






