from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import os
import base64
import pandas as pd
from prophet import Prophet
import yfinance as yf
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
    get_pe_ratio,
    get_52_week_high_low,
    get_market_cap,
    get_next_earnings_date,
)

app = FastAPI()

START = "2000-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

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
    "get_pe_ratio": get_pe_ratio,
    "get_52_week_high_low": get_52_week_high_low,
    "get_market_cap": get_market_cap,
    "get_next_earnings_date": get_next_earnings_date,
}

class FunctionRequest(BaseModel):
    name: str
    params: dict | None = None

@app.post("/function")
def call_function(req: FunctionRequest):
    if req.name not in available_functions:
        raise HTTPException(status_code=404, detail="Function not found")
    try:
        params = req.params or {}
        result = available_functions[req.name](**params)
        if req.name == "plot_stock_price":
            with open("stock.png", "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            return {"result": b64}
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class DataResponse(BaseModel):
    data: list
    info: dict

@app.get("/data/{ticker}", response_model=DataResponse)
def get_data(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=START, end=TODAY)
        info = stock.info
        if data.empty:
            raise HTTPException(status_code=404, detail="No data found")
        data.reset_index(inplace=True)
        df = calculate_indicators(data)
        return {"data": df.to_dict(orient="records"), "info": info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ForecastParams(BaseModel):
    ticker: str
    data_column: str = "Close"
    forecast_years: int = 2

@app.get("/forecast")
def get_forecast(params: ForecastParams):
    try:
        stock = yf.Ticker(params.ticker)
        df = stock.history(start=START, end=TODAY)
        if df.empty:
            raise HTTPException(status_code=404, detail="No data found")
        df.reset_index(inplace=True)
        df_prophet = df[["Date", params.data_column]].dropna().copy()
        df_prophet.columns = ["ds", "y"]
        df_prophet["ds"] = df_prophet["ds"].dt.tz_localize(None)
        model = Prophet(
            changepoint_prior_scale=0.05,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
        )
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=params.forecast_years * 365, freq="D")
        forecast = model.predict(future)
        return {"forecast": forecast.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def calculate_indicators(df: pd.DataFrame, column: str = "Close") -> pd.DataFrame:
    df["MA_20"] = df[column].rolling(window=20).mean()
    df["MA_50"] = df[column].rolling(window=50).mean()
    df["MA_200"] = df[column].rolling(window=200).mean()
    df["BB_Middle"] = df[column].rolling(window=20).mean()
    bb_std = df[column].rolling(window=20).std()
    df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
    df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)
    delta = df[column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    ema12 = df[column].ewm(span=12, adjust=False).mean()
    ema26 = df[column].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df["MACD"] = macd_line
    df["MACD_Signal"] = signal_line
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    return df
