# Define dates for creating the API url:

euroCode <- "EUR"
dollarCode <- "USD"

# Today's date
todaysDate <- now()

oneYearBeforeDate <- todaysDate - years(1) 

getReversedDate <- function(date) {
  
  day <- ""
  month <- ""
  
  if(day(date) <= 9) {
    day <- stringr::str_interp("0${day(date)}")
  } else {
    day <- day(date)
  }
  
  if(month(date) <= 9) {
    month <- stringr::str_interp("0${day(date)}")
  } else {
    month <- month(date)
  }
  
  finalDate <- stringr::str_interp("${day}-${month}-${year(date)}")
  
  return(finalDate)
}

normalizedLastYear <- getReversedDate(oneYearBeforeDate)
                                          
normalizedToday <- getReversedDate(todaysDate)

# Get infos from Banco do Brasil API

dollar_series <- 
