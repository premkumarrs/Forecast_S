"""Global constants for lightweight application-wide enumerations."""
from enum import Enum


class ForecastMode(str, Enum):
    """Available forecast experience modes."""
    CLASSIC = "classic"
    EXISTING_FORECAST_NEWS = "existing_forecast_news"


FORECAST_MODE_LABELS = {
    ForecastMode.CLASSIC: "Forecast + News Adjustment",
    ForecastMode.EXISTING_FORECAST_NEWS: "Existing Forecast + News Adjustment",
}
