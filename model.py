"""
ML Models for Demand Forecasting: Prophet + LightGBM + Hybrid
"""
import pandas as pd
import numpy as np
from prophet import Prophet
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Tuple, Dict, List, Optional
import warnings

warnings.filterwarnings("ignore")


class DemandForecaster:
    def __init__(self):
        self.prophet_model: Optional[Prophet] = None
        self.lgbm_model = None
        self.is_trained = False
        self.feature_info: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_prophet(self, df: pd.DataFrame, date_col: str = "date", sales_col: str = "sales") -> Dict:
        """Train Prophet model for time series forecasting."""
        try:
            prophet_df = df[[date_col, sales_col]].rename(columns={date_col: "ds", sales_col: "y"})
            prophet_df = prophet_df.dropna()

            self.prophet_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
            )
            self.prophet_model.fit(prophet_df)

            future = self.prophet_model.make_future_dataframe(periods=30)
            forecast = self.prophet_model.predict(future)
            mae = mean_absolute_error(prophet_df["y"], forecast["yhat"][: len(prophet_df)])

            return {"model": "Prophet", "mae": round(mae, 2), "status": "success"}

        except Exception as e:
            return {"model": "Prophet", "error": str(e), "status": "failed"}

    def train_lightgbm(self, df: pd.DataFrame, feature_info: Dict) -> Dict:
        """Train LightGBM regression model with engineered features."""
        try:
            features = [f for f in feature_info["features"] if f in df.columns]
            if not features:
                return {"model": "LightGBM", "error": "No feature columns found in data.", "status": "failed"}

            X = df[features].copy()
            y = df[feature_info["target_col"]].copy()

            # Drop rows where target is NaN
            mask = y.notna()
            X, y = X[mask], y[mask]

            split_idx = max(1, int(0.8 * len(X)))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            params = {
                "objective": "regression",
                "metric": "rmse",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.9,
                "verbose": -1,
            }

            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

            self.lgbm_model = lgb.train(
                params,
                train_data,
                valid_sets=[valid_data],
                num_boost_round=100,
                callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)],
            )

            # Metrics on test set (if large enough)
            if len(y_test) > 0:
                y_pred = self.lgbm_model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            else:
                # Full data used for training
                y_pred = self.lgbm_model.predict(X_train)
                mae = mean_absolute_error(y_train, y_pred)
                rmse = np.sqrt(mean_squared_error(y_train, y_pred))

            return {
                "model": "LightGBM",
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "status": "success",
            }

        except Exception as e:
            return {"model": "LightGBM", "error": str(e), "status": "failed"}

    def train_all(self, df: pd.DataFrame, feature_info: Dict) -> List[Dict]:
        """Train all models and return results list."""
        self.feature_info = feature_info
        results = []
        results.append(self.train_prophet(df))
        results.append(self.train_lightgbm(df, feature_info))
        self.is_trained = True
        return results

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------

    def forecast(self, df: pd.DataFrame, periods: int = 30, feature_info: Optional[Dict] = None) -> Dict:
        """Generate forecasts from all trained models."""
        if not self.is_trained or self.prophet_model is None:
            raise ValueError("Models must be trained first.")

        fi = feature_info or self.feature_info
        results: Dict = {}

        # --- Prophet ---
        future = self.prophet_model.make_future_dataframe(periods=periods)
        prophet_fc = self.prophet_model.predict(future)
        results["prophet"] = prophet_fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods).reset_index(drop=True)

        # --- LightGBM ---
        if self.lgbm_model is not None and fi is not None:
            try:
                features = [f for f in fi["features"] if f in df.columns]
                # Use last `periods` rows as a proxy for future feature values
                last_rows = df[features].tail(periods).copy()
                if len(last_rows) < periods:
                    # Repeat the last row if not enough history
                    pad = pd.concat([last_rows.iloc[[-1]]] * (periods - len(last_rows)), ignore_index=True)
                    last_rows = pd.concat([last_rows, pad], ignore_index=True)
                results["lgbm"] = self.lgbm_model.predict(last_rows)
            except Exception as e:
                print(f"LightGBM forecast skipped: {e}")

        return results
