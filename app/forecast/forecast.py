import streamlit as st
from app.forecast.data import load_data
from app.forecast.forecast_logic import create_forecast
from app.forecast.plots import plot_raw_data, plot_forecast

# Streamlit Page Configuration
st.set_page_config(page_title="Stock Prophet", layout="wide")

# Sidebar Inputs
stocks = ["MSFT", "AAPL", "NVDA", "GOOG", "AMZN", "META", "TSM", "TSLA"]
selected_stock = st.sidebar.selectbox("Select a stock:", stocks)
n_years = st.sidebar.slider("Years of prediction:", 1, 10, 3)
period = n_years * 365

# Load and Display Data
data = load_data(selected_stock)
st.write(f"### 📊 Current Data for {selected_stock}")
st.write(data.tail(10))

# Raw Data Plot
raw_fig = plot_raw_data(data, selected_stock)
st.plotly_chart(raw_fig, use_container_width=True)

# Forecasting
data.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)
model, forecast = create_forecast(data, period)

# Forecast Plot
forecast_fig = plot_forecast(model, forecast)
st.plotly_chart(forecast_fig, use_container_width=True)

st.markdown("---\n*Stock Prophet* | Designed for traders and investors")
