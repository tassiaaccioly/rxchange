from selenium import webdriver
import bs4
import time
import re

driver = webdriver.Chrome()

todaysDate = str(r.todaysDate).split()[0]

print(todaysDate)

# alternative url that will need some math to get the exchange value
# driver.get("https://wise.com/br/pricing/send-money?sourceAmount=100&sourceCcy=EUR&targetCcy=BRL")

driver.get("https://wise.com/tools/exchange-rate-alerts/?fromCurrency=BRL&toCurrency=EUR")

# the link for getting the USD value
# driver.get("https://wise.com/tools/exchange-rate-alerts/?fromCurrency=BRL&toCurrency=USD")

# wait for backend to answer and load infos
time.sleep(3)

if driver.page_source:
  data = bs4.BeautifulSoup(driver.page_source, features="html.parser")
  
  driver.close()
else:
  # TODO: fix this error status
  print(f"Error: {res.status_code }")

wiseScrape = data.select('span.text-success')
# this line returns something like: 
# [<span class="text-success">0.172162 EUR</span>, <span class="text-success">5.80848 BRL</span>]

wiseRegex = r"\d+\.\d+ [A-Z]{3}"
# gets everything between the > and < so, "0.172162 EUR" for example

wiseQuotas = []

for i in range(len(wiseScrape)):
  wiseScrape[i] = str(wiseScrape[i])
  print(wiseScrape[i])
  # this line searches for the value + exchange currency symbol in the wiseScrape
  # returns, and splits them into two ["value", "currency symbol"]
  tempValue = re.search(wiseRegex, wiseScrape[i]).group().split()
  wiseQuotas.extend(tempValue)





