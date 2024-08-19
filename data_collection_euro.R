# Source helper functions:
source("helper_functions.R")

# # # # # # # # # # # # # # # # # # # # # # #
# 0. Pull database from Banco do Brasil API #
# # # # # # # # # # # # # # # # # # # # # # #

euroCode <- "EUR"

currentExchange <- "EUR"

# Define dates for creating the API url:
# Getting final historical series date:

todaysDate <- now()
# yesterdaysDate <- todaysDate - days(1)

initialDate <- date("2024-04-05")


# Getting initial historical series date:

oneYearBeforeDate <- initialDate - years(1)

# Normalize dates to fit API style

normalizedLastYear <- getNormalizedDate(oneYearBeforeDate, "us")
normalizedYesterday <- getNormalizedDate(todaysDate - days(1), "us")

normalizedYesterday <- "06-28-2024"

# Get historical series from Banco do Brasil API for euro

eur_series <- getHistoricalSeries(euroCode, normalizedLastYear, normalizedYesterday)


# # # # # # # # # # # # #
# 1. Clean the data set #
# # # # # # # # # # # # #

# 1a. Turn the data_frame into a tibble and remove the "paridade" columns

clean_eur_series <- tibble::as_tibble(eur_series[, c(3, 5, 6)])

# 1b. Rename columns
columnNames <- c("buyPrice", "date", "priceType")

names(clean_eur_series) <- columnNames

# 1c. Fix dates to remove timestamps
clean_eur_series$date <- clean_eur_series$date %>% date()

# 1d. Make a separate date tibble to add proper id
eur_series_dates <- clean_eur_series$date

eur_series_dates <- eur_series_dates[!duplicated(eur_series_dates)]

eur_series_dates <- tibble(eur_series_dates) %>%
  mutate(id = row_number()) %>%
  rename(date = eur_series_dates)

# 1e. Get only the opening rows
clean_eur_series_open <- clean_eur_series[clean_eur_series$priceType == "Abertura", ]

## Rename column and remove priceType Column
clean_eur_series_open <- clean_eur_series_open %>%
  rename(openingPrice = buyPrice) %>%
  mutate(priceType = NULL)

## Left join ids to database
clean_eur_series_open <- left_join(clean_eur_series_open, eur_series_dates, by = "date")

# 1f. Get only the closing rows, rename price column and delete other columns
clean_eur_series_close <- clean_eur_series[clean_eur_series$priceType == 'Fechamento', ] %>%
  rename(closingPrice = buyPrice) %>%
  mutate(priceType = NULL)

## Join closing prices in clean_eur_series_open
clean_eur_series_open <- left_join(clean_eur_series_open, clean_eur_series_close, by = "date")

# 1g. Reorganizing the columns
clean_eur_series_open <- clean_eur_series_open %>% select(id, date, openingPrice, closingPrice)

# 1h. Get dailyAveragePrice

## make a database with all base values and correct ids
clean_eur_series_with_id <- left_join(clean_eur_series, clean_eur_series_open, by = "date") %>% select(id, date, buyPrice)

## adds ids to the database
clean_eur_series_avg <- clean_eur_series_with_id %>%
  mutate(dailyAvgPrice = mean(buyPrice), .by = "id") %>%
  mutate(buyPrice = NULL,
         priceType = NULL,
         id = NULL) %>%
  select(date, dailyAvgPrice)

## remove duplicate rows
clean_eur_series_avg <- clean_eur_series_avg[!duplicated(clean_eur_series_avg), ]

## Join intermediary prices in clean_eur_series_open
clean_eur_series_open <- left_join(clean_eur_series_open, clean_eur_series_avg, by = "date")

# 1i. Get high and low prices by day

## select the correct columns, create a high and low columns with values measured by id and remove the `buyPrice` column
clean_eur_series_high_low <- clean_eur_series_with_id %>%
  select(id, date, buyPrice) %>%
  mutate(high = max(buyPrice), low = min(buyPrice), .by = id) %>%
  select(id, date, high, low)

## remove duplicates
clean_eur_series_high_low <- clean_eur_series_high_low[!duplicated(clean_eur_series_high_low),] %>% select(date, high, low)

## add high/low columns to clean_eur_series_open
clean_eur_series_open <- left_join(clean_eur_series_open, clean_eur_series_high_low, by = "date")

# 1j. Save everything into final_eur_series
final_eur_series <- clean_eur_series_open %>% select(day = id, date, open = openingPrice, close = closingPrice, high, low, dailyAvg = dailyAvgPrice)

# # # # # # # # # # # # # # # # # # # # #
# 2. Generate statistics for the values #
# # # # # # # # # # # # # # # # # # # # #

# 2a. Get descriptives for the databases

descritivasEuro <- describe(clean_eur_series$dailyAverageBuy)

descritivasEuro2 <- clean_eur_series %>% summarise(
  standardDeviation = sd(dailyAverageBuy),
  standartError = sd(dailyAverageBuy) / sqrt(length(dailyAverageBuy)),
  variance = var(dailyAverageBuy),
  moda = getMode(dailyAverageBuy),
  mean = mean(dailyAverageBuy),
  máximo = max(dailyAverageBuy),
  mínimo = min(dailyAverageBuy),
  elementsUnderMean = length(which(dailyAverageBuy < mean(dailyAverageBuy)))
)

length(which(clean_eur_series$dailyAverageBuy < 5.402)) #valores abaixo da média
length(which(clean_eur_series$dailyAverageBuy < 5.362)) #valores abaixo da mediana
length(which(clean_eur_series$dailyAverageBuy < 5.456)) #valores abaixo do terceiro quartil

# 2b. Generate moving averages for


# # # # # # # # # # # # # #
# 3. Calculating the risk #
# # # # # # # # # # # # # #

# 3a.

# # # # # # # # # # # # # # #
# 4. Plot necessary graphs  #
# # # # # # # # # # # # # # #

# 4a. Plot the line graph for clean_eur_series

eur_plot <- ggplot(clean_eur_series) +
  aes(x = date, y = dailyAverageBuy) +
  geom_line(linewidth = 1L, colour = "black") +
  scale_x_date(breaks = scales::breaks_pretty(n = 6)) +
  labs(
    x = "Mês",
    y = "Média diária real/euro (compra)",
    title = "",
    # "Série histórica de taxa de câmbio",
    subtitle = "",
    # stringr::str_interp("${currentExchange} (compra)"),
    caption = "" # stringr::str_interp("${normalizedLastYear} - ${normalizedYesterday}")
  ) +
  geom_line(
    aes(y = mean(buyPrice)),
    colour = "red",
    linewidth = 0.8,
    linetype = "dashed",
    show.legend = TRUE
  ) +
  theme_bw() +
  theme(
    axis.text.y = element_text(size = 11L, colour = "black"),
    axis.text.x = element_text(size = 11L, colour = "black"),
    axis.line = element_line(colour = "black", linewidth = 1.5),
    text = element_text(size = 13L)
  )

eur_plot + lims(y = c(4.7, 6))

# candlestick plot: https://www.r-bloggers.com/2021/09/robservations-12-making-a-candlestick-plot-with-the-ggplot2-and-tidyquant-packages/
