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


# Plot raw data
def plot_raw_data():
    fig = go.Figure()

    # Determine whether the last closing price is higher or lower than the previous one
    increase = data['Close'].iloc[-1] > data['Close'].iloc[-2]
    color = '#32CD32' if increase else '#C70039'
    fillcolor = 'rgba(152, 251, 152, 0.5)' if increase else 'rgba(255, 192, 203, 0.5)'

    # Add a trace with the determined color and reduced line thickness
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close",
                             line=dict(color=color, width=1),
                             fill='tozeroy', fillcolor=fillcolor,
                             mode='lines+markers', marker=dict(size=1)))

    # Add layout with a professional financial theme
    fig.update_layout(
        title_text='Time Series data with Rangeslider',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white',  # professional financial theme
        xaxis_rangeslider_visible=True,
        font=dict(
            family="Courier New, monospace",
            size=12,
            color="#7f7f7f"
        ),
        height=600,  # increase the overall height of the plot

    )

    st.plotly_chart(fig)


plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Stock Forecast')
st.write(forecast.tail(10).reset_index(drop=True), width=1000)

st.subheader(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
fig1.update_layout(
    xaxis_title="Date",
    yaxis_title="Closing Price",
)
st.subheader("Forecast Components")
fig2 = m.plot_components(forecast)
st.write(fig2)

