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

driver.get("https://www.c6bank.com.br/conta-internacional-c6-conta-global")

time.sleep(3)

if driver.page_source:
  ## Get usd page and save page code to variable

  buttonUsd = driver.find_element(By.ID, 'simulator-tabs-tab-0')

  time.sleep(2)

  buttonUsd.send_keys(Keys.RETURN)

  inputBRL = driver.find_element(By.NAME, "simulator-usd-debit-brl")

  inputBRL.clear()

  inputBRL.send_keys('10000')

  dataUsdC6 = bs4.BeautifulSoup(driver.page_source, features="html.parser")

  ## Get eur page and save page code to variable

  buttonEur = driver.find_element(By.ID, 'simulator-tabs-tab-1')

  buttonEur.send_keys(Keys.RETURN)

  time.sleep(2)

  dataEurC6 = bs4.BeautifulSoup(driver.page_source, features="html.parser")

  driver.close()
else:
  # TODO: fix this error status
  print(f"Error: {res.status_code }")

# wiseScrape = data.select('span.text-success')
