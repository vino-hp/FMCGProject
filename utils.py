"""
Data preprocessing and helper functions for demand forecasting project
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
import os
from pathlib import Path

def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV data into pandas DataFrame with error handling"""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print("❌ File not found. Using sample data.")
        return create_sample_data()
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return create_sample_data()

def create_sample_data() -> pd.DataFrame:
    """Create dummy sales data for demonstration"""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    sales = 150 + 20 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.normal(0, 15, len(dates))
    sales = np.maximum(sales, 50)  # Ensure positive sales
    
    df = pd.DataFrame({
        'date': dates,
        'sales': sales.round(2),
        'product_id': 'PROD001',
        'location': 'Coimbatore'
    })
    return df
def preprocess_data(df: pd.DataFrame, weather_data: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Comprehensive data preprocessing pipeline
    Returns: processed_df, feature_info
    """
    # Step 1: Basic validation
    required_cols = ['date', 'sales']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Step 2: Date handling
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    
    # Step 3: Handle missing sales values
    df['sales'] = df['sales'].ffill().bfill().fillna(df['sales'].mean())
    
    # Step 4: Create time-based features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Step 5: Lag features for LightGBM
    df['sales_lag_1'] = df['sales'].shift(1)
    df['sales_lag_7'] = df['sales'].shift(7)
    df['sales_lag_30'] = df['sales'].shift(30)
    df['sales_rolling_mean_7'] = df['sales'].rolling(window=7).mean()
    df['sales_rolling_std_7'] = df['sales'].rolling(window=7).std()
    
    # Fill lag NaNs
    for col in ['sales_lag_1', 'sales_lag_7', 'sales_lag_30', 'sales_rolling_mean_7', 'sales_rolling_std_7']:
        df[col] = df[col].bfill().fillna(df[col].mean())
    # Step 6: Merge weather data if available
    if weather_data is not None and 'date' in weather_data.columns:
        weather_data['date'] = pd.to_datetime(weather_data['date'])
        df = df.merge(weather_data, on='date', how='left')
        # Fill weather NaNs
        weather_cols = ['temperature', 'humidity']
        for col in weather_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())
    
    feature_info = {
        'date_col': 'date',
        'target_col': 'sales',
        'features': ['month', 'day', 'day_of_week', 'quarter', 'is_weekend', 
                    'sales_lag_1', 'sales_lag_7', 'sales_lag_30', 
                    'sales_rolling_mean_7', 'sales_rolling_std_7']
    }
    
    print(f"✅ Preprocessing complete. Shape: {df.shape}")
    return df, feature_info


def calculate_inventory_metrics(avg_demand: float, lead_time: float, safety_stock: float) -> Dict:
    """Calculate reorder point and inventory recommendations"""
    reorder_point = (avg_demand * lead_time) + safety_stock
    
    return {
        'average_daily_demand': round(avg_demand, 2),
        'reorder_point': round(reorder_point, 2),
        'lead_time': lead_time,
        'safety_stock': safety_stock,
        'recommendation': f"Reorder when inventory <= {round(reorder_point, 2)} units"
    }