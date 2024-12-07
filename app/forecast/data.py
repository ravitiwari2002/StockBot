import yfinance as yf
from datetime import date

START = "2000-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

def load_data(ticker):
    """
    Load stock data using yfinance.
    :param ticker: Stock ticker symbol.
    :return: DataFrame with stock data.
    """
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data
