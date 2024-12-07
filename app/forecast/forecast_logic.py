from prophet import Prophet

def create_forecast(data, periods):
    """
    Generate a forecast using Prophet.
    :param data: DataFrame with 'ds' and 'y' columns.
    :param periods: Number of days to forecast.
    :return: Prophet model and forecast DataFrame.
    """
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return model, forecast
