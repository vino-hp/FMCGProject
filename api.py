"""
OpenWeatherMap API integration for weather data
"""
import requests
import pandas as pd
from typing import Optional, Dict, List
import os
from datetime import datetime, timedelta

class WeatherAPI:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize WeatherAPI client
        Get free API key from: https://openweathermap.org/api
        """
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        self.base_url = "http://api.openweathermap.org/data/2.5"
        self.location = {"lat": 11.0168, "lon": 76.9558, "city": "Coimbatore"}  # Default: Coimbatore
        
    def get_historical_weather(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch historical weather data (Note: OpenWeatherMap historical data is paid)
        This demo uses simulated weather data correlated with sales patterns
        """
        try:
            # For demo purposes, generate realistic weather data
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            np.random.seed(42)
            
            # Simulate temperature (25-35°C for Coimbatore) with seasonal patterns
            temp_base = 30 + 3 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365)
            temperature = np.clip(temp_base + np.random.normal(0, 2, len(dates)), 24, 36)
            
            # Simulate humidity (60-90%)
            humidity = np.clip(75 + 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365 + np.pi/2) + 
                             np.random.normal(0, 5, len(dates)), 55, 95)
            
            weather_df = pd.DataFrame({
                'date': dates,
                'temperature': temperature.round(1),
                'humidity': humidity.round(1)
            })
            
            print(f"✅ Generated demo weather data for {len(weather_df)} days")
            return weather_df
            
        except Exception as e:
            print(f"⚠️ Weather API unavailable, using demo data: {str(e)}")
            return self._create_demo_weather(start_date, end_date)
    
    def _create_demo_weather(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fallback demo weather data"""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        weather_df = pd.DataFrame({
            'date': dates,
            'temperature': 30.0,
            'humidity': 75.0
        })
        return weather_df