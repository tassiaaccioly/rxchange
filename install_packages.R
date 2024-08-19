packages <- c("tidyverse", "httr","jsonlite", "rvest", "stringr", "data.table", "esquisse", "Hmisc", "TTR")

install.packages("tidyverse")
install.packages("httr")
install.packeges("rvest")
install.packages("jsonlite")
install.packages("stringr")
install.packages("data.table")
install.packages("esquisse")
install.packages("fable")
install.packages("Hmisc")
install.packages("TTR")


if(sum(as.numeric(!packages %in% installed.packages())) != 0){
  instalador <- packages[!packages %in% installed.packages()]
  for(i in 1:length(instalador)) {
    install.packages(instalador, dependencies = T)
    break()}
  sapply(packages, require, character = T) 
} else {
  sapply(packages, require, character = T) 
}