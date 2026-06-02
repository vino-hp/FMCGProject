"""
Data preprocessing and helper functions for demand forecasting project
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
import os
import io


# Required CSV columns
REQUIRED_COLUMNS: List[str] = ["date", "sales"]
OPTIONAL_COLUMNS: List[str] = ["product", "inventory"]


def validate_csv_columns(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Check whether df contains the required columns.
    Returns (is_valid: bool, message: str).
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        sample_cols = ", ".join(df.columns.tolist()[:8])
        return (
            False,
            (
                f"Missing required column(s): **{', '.join(missing)}**.\n\n"
                f"Your file has: `{sample_cols}`.\n\n"
                "Expected columns: `date`, `sales` (and optionally `product`, `inventory`)."
            ),
        )
    return True, "OK"


def load_data_from_upload(uploaded_file) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Safely load a Streamlit UploadedFile object into a DataFrame.
    Returns (df_or_None, message).
    """
    try:
        # Read bytes so the stream can be rewound without issues
        raw_bytes = uploaded_file.read()
        if len(raw_bytes) == 0:
            return None, "The uploaded file is empty."

        df = pd.read_csv(io.BytesIO(raw_bytes))

        if df.empty:
            return None, "The CSV file contains no data rows."

        # Strip whitespace from column names
        df.columns = df.columns.str.strip().str.lower()

        is_valid, msg = validate_csv_columns(df)
        if not is_valid:
            return None, msg

        # Parse dates
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        bad_dates = df["date"].isna().sum()
        if bad_dates == len(df):
            return None, "Could not parse any dates in the 'date' column. Use YYYY-MM-DD format."
        if bad_dates > 0:
            df = df.dropna(subset=["date"])

        # Coerce sales to numeric
        df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
        bad_sales = df["sales"].isna().sum()
        if bad_sales == len(df):
            return None, "The 'sales' column contains no numeric values."
        if bad_sales > 0:
            df["sales"] = df["sales"].fillna(df["sales"].median())

        df = df.sort_values("date").reset_index(drop=True)
        return df, f"Loaded {len(df):,} rows × {df.shape[1]} columns."

    except UnicodeDecodeError:
        return None, "File encoding error. Please save your CSV as UTF-8 and re-upload."
    except pd.errors.ParserError as e:
        return None, f"CSV parse error: {e}"
    except Exception as e:
        return None, f"Unexpected error reading file: {e}"


def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV from disk path with error handling (used for sample data)."""
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip().str.lower()
        return df
    except FileNotFoundError:
        return create_sample_data()
    except Exception:
        return create_sample_data()


def create_sample_data() -> pd.DataFrame:
    """Generate realistic demo sales data."""
    dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")
    rng = np.random.default_rng(42)
    n = len(dates)
    sales = (
        150
        + 20 * np.sin(np.arange(n) * 2 * np.pi / 365)
        + rng.normal(0, 15, n)
    )
    sales = np.maximum(sales, 50).round(2)
    return pd.DataFrame(
        {
            "date": dates,
            "sales": sales,
            "product": "PROD001",
            "inventory": np.maximum(200 - sales * 0.3 + rng.normal(0, 10, n), 0).round(0),
        }
    )


def preprocess_data(
    df: pd.DataFrame, weather_data: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Comprehensive preprocessing pipeline.
    Returns: (processed_df, feature_info)
    """
    # 1. Validate
    is_valid, msg = validate_csv_columns(df)
    if not is_valid:
        raise ValueError(msg)

    df = df.copy()

    # 2. Date handling
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # 3. Sales cleaning
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df["sales"] = df["sales"].ffill().bfill().fillna(df["sales"].median())

    # 4. Time-based features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # 5. Lag & rolling features
    for lag in [1, 7, 30]:
        col = f"sales_lag_{lag}"
        df[col] = df["sales"].shift(lag)

    df["sales_rolling_mean_7"] = df["sales"].rolling(7).mean()
    df["sales_rolling_std_7"] = df["sales"].rolling(7).std()

    lag_cols = [
        "sales_lag_1", "sales_lag_7", "sales_lag_30",
        "sales_rolling_mean_7", "sales_rolling_std_7",
    ]
    for col in lag_cols:
        df[col] = df[col].bfill().fillna(df[col].median())

    # 6. Merge weather if available
    if weather_data is not None and "date" in weather_data.columns:
        weather_data = weather_data.copy()
        weather_data["date"] = pd.to_datetime(weather_data["date"])
        df = df.merge(weather_data[["date", "temperature", "humidity"]], on="date", how="left")
        for col in ["temperature", "humidity"]:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

    feature_cols = [
        "month", "day", "day_of_week", "quarter", "is_weekend",
        "sales_lag_1", "sales_lag_7", "sales_lag_30",
        "sales_rolling_mean_7", "sales_rolling_std_7",
    ]
    if "temperature" in df.columns:
        feature_cols += ["temperature", "humidity"]

    feature_info = {
        "date_col": "date",
        "target_col": "sales",
        "features": feature_cols,
    }

    return df, feature_info


def calculate_inventory_metrics(
    avg_demand: float, lead_time: float, safety_stock: float
) -> Dict:
    """Calculate reorder point and inventory recommendations."""
    reorder_point = (avg_demand * lead_time) + safety_stock
    eoq_proxy = round(avg_demand * lead_time * 1.5, 2)
    return {
        "average_daily_demand": round(avg_demand, 2),
        "reorder_point": round(reorder_point, 2),
        "lead_time": lead_time,
        "safety_stock": safety_stock,
        "suggested_order_qty": eoq_proxy,
        "recommendation": (
            f"🔔 Reorder when stock drops to **{round(reorder_point, 0):.0f} units**. "
            f"Suggested order quantity: **{eoq_proxy:.0f} units**."
        ),
    }
