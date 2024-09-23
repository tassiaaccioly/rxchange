from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from __future__ import print_function, unicode_literals
from pprint import pprint
from PyInquirer import prompt, Separator
import bs4
import time
import re
import openpyxl

driver = webdriver.Chrome()

todaysDate = str(r.todaysDate).split()[0]

## Getting the dollar page

driver.get("https://www.nomadglobal.com/cotacoes/dolar")

time.sleep(3)

if driver.page_source:
  dataUsdNomad = bs4.BeautifulSoup(driver.page_source, features="html.parser")
  
else:
  # TODO: fix this error status
  print(f"Error: {res.status_code }")

## Getting the euro page

driver.get("https://www.nomadglobal.com/cotacoes/euro")

time.sleep(3)

if driver.page_source:
  dataEurNomad = bs4.BeautifulSoup(driver.page_source, features="html.parser")

  driver.close()
else:
  # TODO: fix this error status
  print(f"Error: {res.status_code }")
