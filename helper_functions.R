getReversedDate <- function(date) {
  
  day <- ""
  month <- ""
  
  if(day(date) <= 9) {
    day <- stringr::str_interp("0${day(date)}")
  } else {
    day <- day(date)
  }
  
  if(month(date) <= 9) {
    month <- stringr::str_interp("0${month(date)}")
  } else {
    month <- month(date)
  }
  
  finalDate <- stringr::str_interp("${day}-${month}-${year(date)}")
  
  return(finalDate)
}

getHistoricalSeries <- function(currency, initialDate, finalDate) {
  url <- stringr::str_interp("https://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/odata/CotacaoMoedaPeriodo(moeda=@moeda,dataInicial=@dataInicial,dataFinalCotacao=@dataFinalCotacao)?@moeda='${currency}'&@dataInicial='${initialDate}'&@dataFinalCotacao='${finalDate}'&$top=360&$format=json")
  
  res <- GET(url)
  
  if(res$status_code == 200) {
    return(fromJSON(rawToChar(res$content))$value)
  } else {
    print("SOMETHING WENT WRONG")
    return(res$status_code)
  }
  
}