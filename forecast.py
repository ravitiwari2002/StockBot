from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import streamlit as st
import yfinance as yf

START = "2000-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Prophet: Predictive Edge')

stocks = (
'MSFT', 'AAPL', 'NVDA', 'GOOG', 'AMZN', 'META', 'TSM', 'AVGO', 'TSLA', 'TCEHY', 'ASML', 'ORCL', '005930.KS', 'NFLX',
'AMD', 'ADBE', 'CRM', 'QCOM', 'SAP', 'PDD','AMAT','CSCO','BABA','INTU','TXN')
selected_stock = st.selectbox('Choose a Ticker Symbol to Explore Stock Forecasts', stocks)

n_years = st.slider('Years of prediction:', 1, 10)
period = n_years * 365


@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text('Loading Stocks Data...')
data = load_data(selected_stock)
data_load_state.text('Stocks Data has been loaded!')

st.subheader('Current Stock Data')
st.dataframe(data.tail(10).reset_index(drop=True), width=1000)
