"""
OpenWeatherMap API integration for weather data
Fetches real-time and historical weather using city name search
"""
import requests
import pandas as pd
import numpy as np
from typing import Optional, Dict
import os
from datetime import datetime, timedelta


class WeatherAPI:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize WeatherAPI client.
        Get a free API key from: https://openweathermap.org/api
        Store it in a .env file as OPENWEATHER_API_KEY=<your_key>
        """
        self.api_key = api_key or os.getenv("OPENWEATHER_API_KEY", "")
        self.base_url = "https://api.openweathermap.org/data/2.5"

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_current_weather(self, city: str) -> Dict:
        """
        Fetch real-time weather for *city* from OpenWeatherMap.
        Returns a dict with keys: city, temperature, humidity, condition, icon.
        On any failure returns a dict with an 'error' key.
        """
        city = (city or "").strip()
        if not city:
            return {"error": "City name cannot be empty."}

        if not self.api_key:
            return {
                "error": (
                    "OpenWeather API key is missing. "
                    "Add OPENWEATHER_API_KEY=<key> to your .env file."
                )
            }

        url = f"{self.base_url}/weather"
        params = {"q": city, "appid": self.api_key, "units": "metric"}

        try:
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 401:
                return {"error": "Invalid API key. Check OPENWEATHER_API_KEY in your .env file."}
            if response.status_code == 404:
                return {"error": f"City '{city}' not found. Please check the spelling."}
            if response.status_code == 429:
                return {"error": "API rate limit exceeded. Please wait a moment and retry."}
            if not response.ok:
                return {"error": f"Weather API error {response.status_code}: {response.text[:200]}"}

            data = response.json()
            return {
                "city": data.get("name", city),
                "country": data.get("sys", {}).get("country", ""),
                "temperature": round(data["main"]["temp"], 1),
                "feels_like": round(data["main"]["feels_like"], 1),
                "humidity": data["main"]["humidity"],
                "condition": data["weather"][0]["description"].title(),
                "icon": data["weather"][0]["icon"],
                "wind_speed": data.get("wind", {}).get("speed", 0),
                "visibility": round(data.get("visibility", 0) / 1000, 1),
            }

        except requests.exceptions.ConnectionError:
            return {"error": "No internet connection. Please check your network."}
        except requests.exceptions.Timeout:
            return {"error": "Request timed out. Please retry."}
        except KeyError as e:
            return {"error": f"Unexpected API response – missing key: {e}"}
        except Exception as e:
            return {"error": f"Unexpected error: {e}"}

    def get_historical_weather(
        self, start_date: str, end_date: str, city: str = "Coimbatore"
    ) -> Optional[pd.DataFrame]:
        """
        Return a DataFrame of daily temperature & humidity between start_date and end_date.

        OpenWeatherMap historical data requires a paid plan; we generate realistic
        simulated data correlated with Southern Indian climate patterns.
        """
        try:
            dates = pd.date_range(start=start_date, end=end_date, freq="D")
            n = len(dates)
            rng = np.random.default_rng(42)

            phase = np.arange(n) * 2 * np.pi / 365
            temp_base = 30 + 3 * np.sin(phase)
            temperature = np.clip(temp_base + rng.normal(0, 2, n), 20, 42).round(1)

            humid_base = 70 + 12 * np.sin(phase + np.pi / 2)
            humidity = np.clip(humid_base + rng.normal(0, 5, n), 40, 98).round(1)

            return pd.DataFrame({"date": dates, "temperature": temperature, "humidity": humidity})

        except Exception as e:
            print(f"⚠️  Weather simulation failed: {e}")
            return self._create_fallback_weather(start_date, end_date)

    def _create_fallback_weather(self, start_date: str, end_date: str) -> pd.DataFrame:
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        return pd.DataFrame({"date": dates, "temperature": 30.0, "humidity": 70.0})
