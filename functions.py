import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def get_stock_price(ticker):
    return str(yf.Ticker(ticker).history(period='1y').iloc[-1].Close)


def calculate_SMA(ticker, window):
    data = yf.Ticker(ticker).history(period='1y').Close
    return str(data.rolling(window=window).mean().iloc[-1])


def calculate_EMA(ticker, window):
    data = yf.Ticker(ticker).history(period='1y').Close
    return str(data.ewm(span=window, adjust=False).mean().iloc[-1])


def calculate_RSI(ticker):
    data = yf.Ticker(ticker).history(period='1y').Close
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=14 - 1, adjust=False).mean()
    ema_down = down.ewm(com=14 - 1, adjust=False).mean()
    rs = ema_up / ema_down
    return str(100 - (100 / (1 + rs)).iloc[-1])


def calculate_MACD(ticker):
    data = yf.Ticker(ticker).history(period='1y').Close
    short_EMA = data.ewm(span=12, adjust=False).mean()
    long_EMA = data.ewm(span=26, adjust=False).mean()
    MACD = short_EMA - long_EMA
    signal = MACD.ewm(span=9, adjust=False).mean()
    MACD_histogram = MACD - signal
    return f'{MACD[-1]}, {signal[-1]}, {MACD_histogram[-1]}'


def plot_stock_price(ticker):
    data = yf.Ticker(ticker).history(period='1y')
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data.Close)
    plt.title(f'{ticker} Stock Price Over Last 12 Months')
    plt.xlabel('Date')
    plt.ylabel('Stock Price ($)')
    plt.grid(True)
    plt.savefig('stock.png')
    plt.close()


def compare_stock_prices(ticker1, ticker2, period):
    data1 = yf.Ticker(ticker1).history(period=period).Close
    data2 = yf.Ticker(ticker2).history(period=period).Close
    return f"The closing prices for {ticker1} are {data1.tolist()} and for {ticker2} are {data2.tolist()}."


def average_volume(ticker, period):
    data = yf.Ticker(ticker).history(period=period).Volume
    return str(data.mean())


def get_dividend_info(ticker):
    dividends = yf.Ticker(ticker).dividends
    return str(dividends)


def get_stock_news(ticker, num_articles=5):
    news = yf.Ticker(ticker).news[:num_articles]
    return news


def calculate_daily_returns(ticker):
    end = datetime.today().strftime('%Y-%m-%d')
    start = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start, end=end, progress=False)
    data['Daily Return'] = data['Adj Close'].pct_change()
    return {k.strftime('%Y-%m-%d'): (v if pd.notnull(v) else "NaN") for k, v in data['Daily Return'].to_dict().items()}