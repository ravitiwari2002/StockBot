from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
import time

# Constants
START = "2000-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Page Configuration
st.set_page_config(
    page_title="Stock Prophet: Predictive Edge",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Aesthetics
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to right, #1d2671, #c33764);
        color: #FFFFFF;
        font-family: 'Arial', sans-serif;
    }
    .css-1kyxreq, .css-18e3th9 {
        background-color: transparent;
        color: #FFFFFF;
    }
    .css-1v0mbdj {
        background-color: #FFFFFF;
        color: #000000;
    }
    .css-1inwz65 {
        color: #FFD700; /* Gold color for sliders */
    }
    .css-1u1jx5m a {
        color: #FFD700; /* Link color */
    }
    .stButton>button {
        background: linear-gradient(to right, #36d1dc, #5b86e5);
        color: #FFFFFF;
        border: none;
        border-radius: 5px;
        padding: 8px 15px;
        font-size: 16px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
    }
    .st-card {
        background: #2e3b55;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and Sidebar
st.title("📈 Stock Prophet: Predictive Edge")
st.sidebar.header("⚙️ Settings")

# Loading Spinner
with st.spinner("Loading page..."):
    time.sleep(1)

# Ticker Selection
stocks = (
    "MSFT", "AAPL", "NVDA", "GOOG", "AMZN", "META", "TSM", "AVGO",
    "TSLA", "TCEHY", "ASML", "ORCL", "005930.KS", "NFLX", "AMD",
    "ADBE", "CRM", "QCOM", "SAP", "PDD", "AMAT", "CSCO", "BABA",
    "INTU", "TXN"
)
selected_stock = st.sidebar.selectbox(
    "Choose a Ticker Symbol to Explore Stock Forecasts", stocks
)

# Prediction Period
n_years = st.sidebar.slider("Years of prediction:", 1, 10, 3)
period = n_years * 365

# Data Loading
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Loading Stocks Data...")
data = load_data(selected_stock)
data_load_state.text("✅ Stocks Data has been loaded!")

# Show Stock Data
st.markdown("### 📊 Current Stock Data")
st.markdown(
    f"""
    <div class="st-card">
    <p style="font-size: 16px;">The table below shows the latest stock data for <b>{selected_stock}</b>.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write(data.tail(10).reset_index(drop=True))

# Raw Data Plot
def plot_raw_data():
    fig = go.Figure()

    # Determine price increase/decrease for coloring
    color = "#17BECF"  # Use teal for better distinction
    fillcolor = "rgba(23, 190, 207, 0.2)"

    # Add trace with color and smooth lines
    fig.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data["Close"],
            mode="lines+markers",
            name="Closing Price",
            line=dict(color=color, width=2, dash="solid"),
            fill="tozeroy",
            fillcolor=fillcolor,
            marker=dict(size=5),
        )
    )

    # Add layout with range slider and buttons
    fig.update_layout(
        title=dict(
            text=f"📈 {selected_stock} Stock Closing Prices",
            font=dict(size=20, color="#FFD700"),
            x=0.5,
        ),
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.2)",
            color="#FFD700",
            rangeselector=dict(
                buttons=[
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(step="all"),
                ]
            ),
            rangeslider=dict(visible=True),
        ),
        yaxis=dict(
            title="Price",
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.2)",
            color="#FFD700",
        ),
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

plot_raw_data()

# Forecasting with Prophet
df_train = data[["Date", "Close"]]
df_train.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show Forecast Data
st.markdown("### 📅 Stock Forecast Data")
st.markdown(
    """
    <div class="st-card">
    <p style="font-size: 16px;">The forecast data is shown below:</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write(forecast.tail(10).reset_index(drop=True))

# Forecast Plot
st.markdown(f"### 📉 Forecast Plot for {n_years} Years")
fig1 = plot_plotly(m, forecast)
fig1.update_layout(
    title=dict(text=f"Predicted Stock Prices for {selected_stock}", x=0.5),
    xaxis=dict(title="Date", color="#FFD700"),
    yaxis=dict(title="Predicted Price", color="#FFD700"),
    plot_bgcolor="rgba(0, 0, 0, 0)",
    paper_bgcolor="rgba(0, 0, 0, 0)",
)
st.plotly_chart(fig1, use_container_width=True)

# Forecast Components
st.markdown("### 🔍 Forecast Components")
st.markdown(
    """
    <div class="st-card">
    <p style="font-size: 16px;">Explore the trend, weekly, and yearly components of the forecast for better insights.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

components = [
    ("Trend", forecast["ds"], forecast["trend"], "Trend"),
    ("Weekly", forecast["ds"], forecast["weekly"], "Weekly Seasonality"),
    ("Yearly", forecast["ds"], forecast["yearly"], "Yearly Seasonality"),
]

for component_name, x, y, title in components:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=component_name))
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16, color="#FFD700")),
        xaxis=dict(title="Date", color="#FFD700"),
        yaxis=dict(title=component_name, color="#FFD700"),
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
    )
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown(
    """
    ---
    *Stock Prophet* | Designed with ❤️ for traders and investors
    """
)
