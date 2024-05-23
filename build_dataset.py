from selenium import webdriver
import bs4
import time
import re

driver = webdriver.Chrome()

# driver.implicitly_wait(10)

# alternative url that will need some math to get the exchange value
# driver.get("https://wise.com/br/pricing/send-money?sourceAmount=100&sourceCcy=EUR&targetCcy=BRL")

driver.get("https://wise.com/tools/exchange-rate-alerts/?fromCurrency=BRL&toCurrency=EUR")

# wait for backend to answer and load infos
time.sleep(3)

if res.status_code == 200:
  data = bs4.BeautifulSoup(driver.page_source)
  
  driver.close()
else:
    print(f"Error: {res.status_code}")

wiseScrape = data.select('span.text-success')

valueRegex = re.compile(r'>(\d.+)<')

for i in wiseScrape:
  value[i] = valueRegex.search(str(wiseScrape[i]))


