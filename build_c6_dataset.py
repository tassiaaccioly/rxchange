from selenium import webdriver
from __future__ import print_function, unicode_literals
from pprint import pprint
from PyInquirer import prompt, Separator
import bs4
import time
import re
import openpyxl

driver = webdriver.Chrome()

todaysDate = str(r.todaysDate).split()[0]

# https://www.c6bank.com.br/conta-internacional-c6-conta-global

driver.get("https://www.c6bank.com.br/conta-internacional-c6-conta-global")

time.sleep(3)

if driver.page_source:
  data = bs4.BeautifulSoup(driver.page_source, features="html.parser")
  
  driver.close()
else:
  # TODO: fix this error status
  print(f"Error: {res.status_code }")

# wiseScrape = data.select('span.text-success')
