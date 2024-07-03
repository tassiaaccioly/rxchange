# # # # # # # # # # # # # # # # # # # # # # # # # #
# Turns dates in any format into DD-MM-YYYY dates #
#             or MM-DD-YYYY dates                 #
# # # # # # # # # # # # # # # # # # # # # # # # # #

getNormalizedDate <- function(date, style) {
  day <- ""
  month <- ""
  
  if (day(date) <= 9) {
    day <- stringr::str_interp("0${day(date)}")
  } else {
    day <- day(date)
  }
  
  if (month(date) <= 9) {
    month <- stringr::str_interp("0${month(date)}")
  } else {
    month <- month(date)
  }
  
  if (style == "us") {
    return(stringr::str_interp("${month}-${day}-${year(date)}"))
  } else {
    return(stringr::str_interp("${day}-${month}-${year(date)}"))
  }
}

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  Calls the Olinda (BACEN) API with arguments and returns  #
#     the historical series for the specified currency      #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

getHistoricalSeries <- function(currency, initialDate, finalDate) {
  url <- stringr::str_interp(
    "https://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/odata/CotacaoMoedaPeriodo(moeda=@moeda,dataInicial=@dataInicial,dataFinalCotacao=@dataFinalCotacao)?@moeda='${currency}'&@dataInicial='${initialDate}'&@dataFinalCotacao='${finalDate}'&$top=9999&$format=json"
  )
  
  res <- GET(url)
  
  if (res$status_code == 200) {
    return(fromJSON(rawToChar(res$content))$value)
  } else {
    print("SOMETHING WENT WRONG")
    return(res$status_code)
  }
  
}

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  Returns the correct foreign exchange for the given exchange code #
#     & the correct exchange code for the given foreign exchange    #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

exchangeMatch <- tibble(code = c("EUR", "USD"),
                        coin = c("euro", "dollar"))

fromCodeToCoin <- function(code) {
  index <- match(code, exchangeMatch$code)
  return(exchangeMatch$coin[index])
}

fromCoinToCode <- function(coin) {
  index <- match(coin, exchangeMatch$coin)
  return(exchangeMatch$code[index])
}

# # # # # # # # # # # # # # # # # # # # # # # # # #
# Returns the mode for a specified numeric vector #
# # # # # # # # # # # # # # # # # # # # # # # # # #

getMode <- function(x) {
  u <- unique(x)
  tab <- tabulate(match(x, u))
  u[tab == max(tab)]
}