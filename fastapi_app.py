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
    openai.api_key = f.read().strip()

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
    response = openai.ChatCompletion.create(
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
            second = openai.ChatCompletion.create(model=MODEL_NAME, messages=messages)
            final_msg = second["choices"][0]["message"]["content"]
            return {"response": final_msg}
    return {"response": response_message.get("content", "")}

@app.get("/api/forecast")
async def forecast(symbol: str, years: int = 1):
    data = yf.Ticker(symbol).history(period="max")
    data = data.reset_index()[["Date", "Close"]]
    data.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)
    m = Prophet()
    m.fit(data)
    future = m.make_future_dataframe(periods=365 * years)
    forecast = m.predict(future)
    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(365 * years)
    return {"forecast": result.to_dict(orient="records")}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
