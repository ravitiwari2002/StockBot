import json
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import openai
import yfinance as yf
from prophet import Prophet
import pandas as pd
from datetime import date
from functions import (
    get_stock_price,
    calculate_SMA,
    calculate_EMA,
    calculate_RSI,
    calculate_MACD,
    plot_stock_price,
    compare_stock_prices,
    average_volume,
    get_dividend_info,
    get_stock_news,
    calculate_daily_returns,
)

app = FastAPI()

# Serve static assets and frontend files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("frontend/index.html")

# Load API key and function config
with open("API_KEY", "r") as f:
    key_line = f.read().strip()
    openai.api_key = key_line.split("=", 1)[-1] if "=" in key_line else key_line

with open("function_config.json", "r") as f:
    config_data = json.load(f)

available_functions = {
    "get_stock_price": get_stock_price,
    "calculate_SMA": calculate_SMA,
    "calculate_EMA": calculate_EMA,
    "calculate_RSI": calculate_RSI,
    "calculate_MACD": calculate_MACD,
    "plot_stock_price": plot_stock_price,
    "compare_stock_prices": compare_stock_prices,
    "average_volume": average_volume,
    "get_dividend_info": get_dividend_info,
    "get_stock_news": get_stock_news,
    "calculate_daily_returns": calculate_daily_returns,
}

MODEL_NAME = "gpt-3.5-turbo-0125"

@app.post("/api/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message", "")
    messages = [{"role": "user", "content": user_message}]
    response = openai.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        functions=config_data,
        function_call="auto",
    )
    response_message = response["choices"][0]["message"]

    if response_message.get("function_call"):
        fn_name = response_message["function_call"]["name"]
        args = json.loads(response_message["function_call"].get("arguments", "{}"))
        fn = available_functions.get(fn_name)
        if fn:
            result = fn(**args)
            messages.append(response_message)
            messages.append({"role": "function", "name": fn_name, "content": result})
            second = openai.chat.completions.create(model=MODEL_NAME, messages=messages)
            final_msg = second["choices"][0]["message"]["content"]
            return {"response": final_msg}
    return {"response": response_message.get("content", "")}

@app.get("/api/forecast")
async def forecast(symbol: str, years: int = 1, column: str = "Close"):
    data = yf.Ticker(symbol).history(period="max")
    if data.empty:
        return {"error": "No data found"}

    data.index = data.index.tz_localize(None)
    data.reset_index(inplace=True)
    data["Date"] = pd.to_datetime(data["Date"]).dt.tz_localize(None)

    # Technical indicators
    data["MA_20"] = data[column].rolling(window=20).mean()
    data["MA_50"] = data[column].rolling(window=50).mean()
    data["MA_200"] = data[column].rolling(window=200).mean()

    data["BB_Middle"] = data[column].rolling(window=20).mean()
    bb_std = data[column].rolling(window=20).std()
    data["BB_Upper"] = data["BB_Middle"] + (bb_std * 2)
    data["BB_Lower"] = data["BB_Middle"] - (bb_std * 2)

    delta = data[column].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))

    ema12 = data[column].ewm(span=12, adjust=False).mean()
    ema26 = data[column].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    data["MACD"] = macd_line
    data["MACD_Signal"] = signal_line
    data["MACD_Hist"] = data["MACD"] - data["MACD_Signal"]

    hist_json = data.to_dict(orient="records")

    prophet_df = data[["Date", column]].dropna().rename(columns={"Date": "ds", column: "y"})
    prophet_df["ds"] = prophet_df["ds"].dt.tz_localize(None)

    m = Prophet(changepoint_prior_scale=0.05, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, seasonality_mode="multiplicative")
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=365 * years)
    forecast = m.predict(future)

    forecast_subset = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(365 * years)

    current_price = data[column].iloc[-1]
    prev_price = data[column].iloc[-2]
    price_change = ((current_price - prev_price) / prev_price) * 100
    predicted_price = forecast_subset["yhat"].iloc[-1]
    forecast_change = ((predicted_price - current_price) / current_price) * 100
    trend_slope = (forecast["trend"].iloc[-1] - forecast["trend"].iloc[-365]) / 365 if len(forecast) > 365 else 0
    trend_direction = "📈 Bullish" if trend_slope > 0 else "📉 Bearish"
    uncertainty = forecast_subset["yhat_upper"].iloc[-1] - forecast_subset["yhat_lower"].iloc[-1]
    confidence = max(0.0, 100 - (uncertainty / predicted_price * 100)) if predicted_price else 0.0

    metrics = {
        "current_price": float(current_price),
        "price_change": float(price_change),
        "predicted_price": float(predicted_price),
        "forecast_change": float(forecast_change),
        "trend_direction": trend_direction,
        "confidence": float(confidence),
    }

    return {
        "historical_data": hist_json,
        "forecast_data": forecast_subset.to_dict(orient="records"),
        "metrics": metrics,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
