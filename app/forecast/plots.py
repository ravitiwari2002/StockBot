import plotly.graph_objects as go
from prophet.plot import plot_plotly

def plot_raw_data(data, stock_name):
    """
    Plot raw stock data with dynamic coloring.
    :param data: DataFrame with stock data.
    :param stock_name: Stock ticker symbol.
    :return: Plotly Figure object.
    """
    data["Change"] = data["Close"].diff()
    colors = data["Change"].apply(lambda x: "#00FF00" if x >= 0 else "#FF0000")  # Green for profit, red for loss

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data["Close"],
            mode="lines+markers",
            name="Closing Price",
            line=dict(color="#FFFFFF", width=2),
            marker=dict(size=7, color=colors),
        )
    )
    fig.update_layout(
        title=f"📈 {stock_name} Stock Closing Prices",
        xaxis_title="Date",
        yaxis_title="Price",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
    )
    return fig

def plot_forecast(model, forecast):
    """
    Plot forecast using Prophet's plotly integration.
    :param model: Trained Prophet model.
    :param forecast: Forecast DataFrame.
    :return: Plotly Figure object.
    """
    fig = plot_plotly(model, forecast)
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Predicted Price",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
    )
    return fig
