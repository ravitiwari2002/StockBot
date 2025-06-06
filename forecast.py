from datetime import date
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & GLOBALS
# ─────────────────────────────────────────────────────────────────────────────
START = "2000-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION & CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Prophet - AI Forecasting",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    body, .stApp, .main {
        background: #1e1e1e;  /* solid dark background across entire app */
        font-family: 'Inter', sans-serif;
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(5px);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .title-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(5px);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(5px);
    }

    .stSelectbox > div > div > div {
        background: rgba(255, 255, 255, 0.05);
        color: white;
        border-radius: 8px;
    }

    .stButton > button {
        background: linear-gradient(45deg, #00CC66, #FF6600);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 10px 24px;
        font-weight: 500;
        font-size: 14px;
        transition: all 0.2s ease;
        box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.5);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px 0 rgba(0, 0, 0, 0.7);
    }

    h1, h2, h3 {
        color: #FFFFFF;
        font-weight: 700;
    }

    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(5px);
        border-radius: 8px;
        padding: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    table.dataframe td, table.dataframe th {
        color: #fff;
        background-color: rgba(30,30,30,0.8);
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TITLE SECTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-container">
    <h1 style="font-size: 2.8rem; margin-bottom: 8px;">📈 Stock Prophet</h1>
    <p style="font-size: 1rem; color: rgba(255,255,255,0.8); margin: 0;">
        AI-Powered Stock Forecasting & Technical Analysis
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## ⚙️ Configuration")

popular_stocks = {
    "🏢 Tech Giants": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX"],
    "💰 Finance": ["JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC"],
    "🏭 Industrial": ["GE", "BA", "CAT", "MMM", "HON", "UPS", "FDX", "LMT"],
    "🏥 Healthcare": ["JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "ABT", "DHR"],
    "🛒 Consumer": ["PG", "KO", "PEP", "WMT", "HD", "MCD", "NKE", "SBUX"],
    "🏦 Crypto & Fintech": ["COIN", "SQ", "PYPL", "HOOD", "SOFI", "AFRM", "LC", "UPST"]
}

all_stocks = [ticker for sublist in popular_stocks.values() for ticker in sublist]

selected_stock = st.sidebar.selectbox(
    "🎯 Select Stock Symbol",
    options=all_stocks,
    index=0,
    help="Choose a stock to analyze and forecast"
)

st.sidebar.markdown("### 📊 Analysis Settings")
data_column = st.sidebar.selectbox(
    "Price Data",
    ["Close", "Open", "High", "Low"],
    index=0,
    help="Select which price data to analyze"
)

forecast_years = st.sidebar.slider(
    "Forecast Period (Years)",
    min_value=1,
    max_value=10,
    value=2,
    help="Number of years to forecast into the future"
)

st.sidebar.markdown("### 🔧 Technical Analysis Options")
show_ma = st.sidebar.checkbox("Moving Averages", value=True)
show_bollinger = st.sidebar.checkbox("Bollinger Bands", value=False)
show_rsi = st.sidebar.checkbox("RSI Indicator", value=False)
show_macd = st.sidebar.checkbox("MACD Indicator", value=True)
show_volume = st.sidebar.checkbox("Volume Analysis", value=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & INDICATOR CALCULATIONS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=START, end=TODAY)
        info = stock.info
        if data.empty:
            return None, None
        data.reset_index(inplace=True)
        return data, info
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {e}")
        return None, None


def calculate_technical_indicators(df):
    # MOVING AVERAGES
    df['MA_20'] = df[data_column].rolling(window=20).mean()
    df['MA_50'] = df[data_column].rolling(window=50).mean()
    df['MA_200'] = df[data_column].rolling(window=200).mean()

    # BOLLINGER BANDS
    df['BB_Middle'] = df[data_column].rolling(window=20).mean()
    bb_std = df[data_column].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)

    # RSI (14‐day)
    delta = df[data_column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD (12‐26 EMA) + Signal (9 EMA of MACD) + Histogram
    ema12 = df[data_column].ewm(span=12, adjust=False).mean()
    ema26 = df[data_column].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df['MACD'] = macd_line
    df['MACD_Signal'] = signal_line
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    return df


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA FOR SELECTED STOCK
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner(f"🔄 Loading data for {selected_stock}..."):
    stock_data, stock_info = load_stock_data(selected_stock)

if stock_data is None:
    st.error("❌ Unable to load stock data. Please try a different symbol.")
    st.stop()

stock_data = calculate_technical_indicators(stock_data)

# ─────────────────────────────────────────────────────────────────────────────
# TOP METRICS ROW
# ─────────────────────────────────────────────────────────────────────────────
if stock_info:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        current_price = stock_data[data_column].iloc[-1]
        prev_price = stock_data[data_column].iloc[-2]
        change = current_price - prev_price
        change_pct = (change / prev_price) * 100
        st.metric(
            label=f"💲 Current {data_column}",
            value=f"${current_price:.2f}",
            delta=f"{change:.2f} ({change_pct:.2f}%)",
            delta_color="normal"
        )
    with col2:
        high_52w = stock_data[data_column].rolling(252).max().iloc[-1]
        st.metric(label="📈 52W High", value=f"${high_52w:.2f}")
    with col3:
        low_52w = stock_data[data_column].rolling(252).min().iloc[-1]
        st.metric(label="📉 52W Low", value=f"${low_52w:.2f}")
    with col4:
        avg_volume = stock_data['Volume'].rolling(30).mean().iloc[-1]
        st.metric(label="📊 Avg Volume (30d)", value=f"{avg_volume/1e6:.1f}M")

# ─────────────────────────────────────────────────────────────────────────────
# SHOW PRICE STATISTICS TABLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 📑 Price Statistics")
stats_df = pd.DataFrame({
    "Metric": ["Mean", "Median", "Std Dev", "Min", "Max"],
    "Value": [
        stock_data[data_column].mean(),
        stock_data[data_column].median(),
        stock_data[data_column].std(),
        stock_data[data_column].min(),
        stock_data[data_column].max()
    ]
})
stats_df["Value"] = stats_df["Value"].map(lambda x: f"${x:.2f}")
st.dataframe(stats_df.style.set_properties(
    **{"background-color": "#2a2a2a", "color": "#ffffff"}
), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PRICE & INDICATORS CHART
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## 📈 Stock Price & Indicators")

# Determine how many subplots to show (candlestick always + any combination of RSI, MACD, Volume)
n_rows = 1
if show_rsi and show_volume and show_macd:
    n_rows = 4
elif (show_rsi and show_volume) or (show_rsi and show_macd) or (show_volume and show_macd):
    n_rows = 3
elif show_rsi or show_volume or show_macd:
    n_rows = 2

row_heights = []
subplot_titles = []
# Row 1: Candlestick + MA
row_heights.append(0.5)
subplot_titles.append(f"{selected_stock} Candlestick + MAs")
# Row 2: RSI (optional)
if show_rsi:
    row_heights.append(0.2)
    subplot_titles.append("RSI (14)")
# Row 3: MACD (optional)
if show_macd:
    row_heights.append(0.2)
    subplot_titles.append("MACD")
# Row 4: Volume (optional)
if show_volume:
    row_heights.append(0.2)
    subplot_titles.append("Volume")

fig = make_subplots(
    rows=n_rows,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=row_heights,
    subplot_titles=subplot_titles
)

# 1) CANDLESTICK + MOVING AVERAGES
fig.add_trace(
    go.Candlestick(
        x=stock_data['Date'],
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        name="Price",
        increasing_line_color="#00FF00",  # green
        decreasing_line_color="#FF0000",  # red
        hoverinfo="x+y+name"              # default hover info (no hovertemplate)
    ),
    row=1, col=1
)

if show_ma:
    # MA20 (orange), MA50 (green), MA200 (red)
    for ma, color in zip([20, 50, 200], ['#FFA500', '#00FF00', '#FF0000']):
        fig.add_trace(
            go.Scatter(
                x=stock_data['Date'],
                y=stock_data[f"MA_{ma}"],
                mode="lines",
                name=f"MA{ma}",
                line=dict(color=color, width=1.5),
                hovertemplate=f"MA{ma}: $%{{y:.2f}}<extra></extra>"
            ),
            row=1, col=1
        )

# Add rangeslider for zoom/pan on the x-axis
fig.update_layout(
    xaxis=dict(
        rangeslider=dict(visible=True, thickness=0.05, bgcolor="#333333"),
        showgrid=False,
        showline=True,
        linecolor="#444444",
        tickfont=dict(color="#DDD")
    ),
    yaxis=dict(
        showgrid=False,
        showline=True,
        linecolor="#444444",
        tickfont=dict(color="#DDD")
    )
)

# 2) RSI (if enabled)
if show_rsi:
    fig.add_trace(
        go.Scatter(
            x=stock_data['Date'],
            y=stock_data['RSI'],
            mode="lines",
            name="RSI",
            line=dict(color="#FFA500", width=1.5),  # orange
            hovertemplate="RSI: %{y:.2f}<extra></extra>"
        ),
        row=2, col=1
    )
    fig.add_hline(
        y=70,
        line_dash="dash",
        line_color="red",
        opacity=0.5,
        row=2, col=1
    )
    fig.add_hline(
        y=30,
        line_dash="dash",
        line_color="green",
        opacity=0.5,
        row=2, col=1
    )

# 3) MACD (if enabled)
if show_macd:
    macd_row = 2 if not show_rsi else 3
    fig.add_trace(
        go.Scatter(
            x=stock_data['Date'],
            y=stock_data['MACD'],
            mode="lines",
            name="MACD Line",
            line=dict(color="#00FF00", width=1.5),  # green
            hovertemplate="MACD: %{y:.2f}<extra></extra>"
        ),
        row=macd_row, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=stock_data['Date'],
            y=stock_data['MACD_Signal'],
            mode="lines",
            name="Signal Line",
            line=dict(color="#FFA500", width=1),  # orange
            hovertemplate="Signal: %{y:.2f}<extra></extra>"
        ),
        row=macd_row, col=1
    )
    fig.add_trace(
        go.Bar(
            x=stock_data['Date'],
            y=stock_data['MACD_Hist'],
            name="Histogram",
            marker_color=np.where(stock_data['MACD_Hist'] >= 0, "#00FF00", "#FF0000"),  # green/red
            hovertemplate="Hist: %{y:.2f}<extra></extra>"
        ),
        row=macd_row, col=1
    )

# 4) VOLUME (if enabled)
if show_volume:
    vol_row = n_rows
    colors = np.where(stock_data['Close'] >= stock_data['Open'], "#00FF00", "#FF0000")
    fig.add_trace(
        go.Bar(
            x=stock_data['Date'],
            y=stock_data['Volume'],
            name="Volume",
            marker_color=colors,
            hovertemplate="Volume: %{y:,}<extra></extra>"
        ),
        row=vol_row, col=1
    )

# FINAL LAYOUT TWEAKS
fig.update_layout(
    template="plotly_dark",
    showlegend=True,
    height=650 + (n_rows - 1) * 150,
    margin=dict(t=40, b=40, l=40, r=40),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1)
)

for i in range(1, n_rows + 1):
    fig.update_xaxes(showgrid=False, row=i, col=1, tickfont=dict(color="#DDD"), linecolor="#444444")
    fig.update_yaxes(showgrid=False, row=i, col=1, tickfont=dict(color="#DDD"), linecolor="#444444")

st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# FORECASTING SECTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## 🔮 AI Forecasting")

# Prepare data for Prophet
df_prophet = stock_data[['Date', data_column]].copy()
df_prophet.columns = ['ds', 'y']
df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
df_prophet = df_prophet.dropna()

if len(df_prophet) < 100:
    st.warning("⚠️ Insufficient data for reliable forecasting. Need at least 100 data points.")
else:
    with st.spinner("🤖 Generating AI forecast..."):
        try:
            model = Prophet(
                changepoint_prior_scale=0.05,
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='multiplicative'
            )
            model.fit(df_prophet)

            future = model.make_future_dataframe(periods=forecast_years * 365, freq="D")
            forecast = model.predict(future)

            # METRICS CARDS
            col1, col2, col3 = st.columns(3)
            with col1:
                last_actual = df_prophet['y'].iloc[-1]
                future_price = forecast['yhat'].iloc[-1]
                price_change = future_price - last_actual
                price_change_pct = (price_change / last_actual) * 100
                st.metric(
                    label=f"Predicted Price ({forecast_years}Y)",
                    value=f"${future_price:.2f}",
                    delta=f"{price_change:.2f} ({price_change_pct:.1f}%)"
                )
            with col2:
                trend_slope = (forecast['trend'].iloc[-1] - forecast['trend'].iloc[-365]) / 365
                trend_direction = "📈 Bullish" if trend_slope > 0 else "📉 Bearish"
                st.metric(label="Trend Direction", value=trend_direction)
            with col3:
                uncertainty = forecast['yhat_upper'].iloc[-1] - forecast['yhat_lower'].iloc[-1]
                confidence = max(0, 100 - (uncertainty / future_price * 100))
                st.metric(label="Confidence Level", value=f"{confidence:.1f}%")

            # FORECAST CHART
            fig_f = go.Figure()
            # Historical (green)
            fig_f.add_trace(
                go.Scatter(
                    x=df_prophet['ds'],
                    y=df_prophet['y'],
                    mode="lines",
                    name="Historical",
                    line=dict(color="#00FF00", width=2),
                    hovertemplate="Historical: $%{y:.2f}<extra></extra>"
                )
            )
            # Forecast (orange dashed)
            fig_f.add_trace(
                go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat'],
                    mode="lines",
                    name="Forecast",
                    line=dict(color="#FFA500", width=2, dash="dash"),
                    hovertemplate="Forecast: $%{y:.2f}<extra></extra>"
                )
            )
            # Upper & lower bounds (semi‐transparent red)
            fig_f.add_trace(
                go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat_upper'],
                    mode="lines",
                    name="Upper Bound",
                    line=dict(color="rgba(255,0,0,0.2)"),
                    showlegend=False
                )
            )
            fig_f.add_trace(
                go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat_lower'],
                    mode="lines",
                    name="Lower Bound",
                    line=dict(color="rgba(255,0,0,0.2)"),
                    fill="tonexty",
                    fillcolor="rgba(255,0,0,0.1)",
                    showlegend=False
                )
            )

            fig_f.update_layout(
                title=f"{selected_stock} Price Forecast ‒ Next {forecast_years} Years",
                template="plotly_dark",
                height=500,
                xaxis_title="Date",
                yaxis_title=f"Price ($)",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=40, b=40, l=40, r=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig_f.update_xaxes(showgrid=False, linecolor="#444444", tickfont=dict(color="#DDD"))
            fig_f.update_yaxes(showgrid=False, linecolor="#444444", tickfont=dict(color="#DDD"))

            st.plotly_chart(fig_f, use_container_width=True)

            # FORECAST COMPONENTS (Trend / Yearly / Weekly)
            st.markdown("### 🔍 Forecast Components Analysis")
            comp = make_subplots(
                rows=3, cols=1,
                subplot_titles=("Trend", "Yearly Seasonality", "Weekly Seasonality"),
                vertical_spacing=0.08
            )
            comp.add_trace(
                go.Scatter(
                    x=forecast['ds'],
                    y=forecast['trend'],
                    mode="lines",
                    name="Trend",
                    line=dict(color="#00FF00", width=2),
                    hovertemplate="Trend: $%{y:.2f}<extra></extra>"
                ),
                row=1, col=1
            )
            comp.add_trace(
                go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yearly'],
                    mode="lines",
                    name="Yearly",
                    line=dict(color="#FFA500", width=2),
                    hovertemplate="Yearly: %{y:.2f}<extra></extra>"
                ),
                row=2, col=1
            )
            comp.add_trace(
                go.Scatter(
                    x=forecast['ds'],
                    y=forecast['weekly'],
                    mode="lines",
                    name="Weekly",
                    line=dict(color="#FF0000", width=2),
                    hovertemplate="Weekly: %{y:.2f}<extra></extra>"
                ),
                row=3, col=1
            )
            comp.update_layout(
                template="plotly_dark",
                height=750,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=40, b=40, l=40, r=40),
                showlegend=False
            )
            for r in range(1, 4):
                comp.update_xaxes(showgrid=False, linecolor="#444444", tickfont=dict(color="#DDD"), row=r, col=1)
                comp.update_yaxes(showgrid=False, linecolor="#444444", tickfont=dict(color="#DDD"), row=r, col=1)

            st.plotly_chart(comp, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Forecasting error: {str(e)}")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY CARDS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## 💡 Key Insights")

c1, c2 = st.columns(2)
with c1:
    st.markdown("""
    <div class="feature-card">
        <h3>📊 Technical Analysis</h3>
        <ul style="margin:0; padding-left:1rem; color:#DDD;">
            <li>Moving averages (MA20, MA50, MA200) reveal trend direction.</li>
            <li>RSI shows overbought/oversold zones (<b>70</b>/<b>30</b>).</li>
            <li>MACD histogram highlights momentum shifts.</li>
            <li>Volume bars colored green/red confirm price moves.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="feature-card">
        <h3>🔮 AI Forecast</h3>
        <ul style="margin:0; padding-left:1rem; color:#DDD;">
            <li>Prophet model captures trend & seasonality.</li>
            <li>Confidence bands gauge prediction uncertainty.</li>
            <li>Component plots (Trend/Yearly/Weekly) show periodic effects.</li>
            <li>Use rangeslider to zoom in on any date range.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.5); padding: 16px;">
    📈 <strong>Stock Prophet</strong> ‒ AI-Powered Financial Analysis  
    ⚠️ This is for educational purposes only. Not financial advice. Always do your own research.
</div>
""", unsafe_allow_html=True)
