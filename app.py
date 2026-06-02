"""
Streamlit Dashboard – Demand Forecasting & Inventory Optimization
Fixed version: proper CSV upload, weather API, error handling, session state
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
from datetime import timedelta

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed – env vars must be set externally

from utils import (
    load_data,
    load_data_from_upload,
    create_sample_data,
    preprocess_data,
    calculate_inventory_metrics,
)
from model import DemandForecaster
from api import WeatherAPI
from auth import require_auth, logout, is_authenticated

# ──────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="FMCG Demand Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Session-state initialisation
# ──────────────────────────────────────────────
_defaults = {
    "df": None,
    "processed_df": None,
    "feature_info": None,
    "forecaster": None,
    "trained": False,
    "weather_data": None,
    "selected_city": "Coimbatore",
}
for key, val in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-header{font-size:2.4rem;color:#1f77b4;text-align:center;margin-bottom:1.5rem;}
    .weather-card{background:linear-gradient(135deg,#1a73e8,#0d47a1);padding:1.2rem;
                  border-radius:12px;color:white;margin-bottom:1rem;}
    .weather-card h3{margin:0;font-size:1.1rem;opacity:.85;}
    .weather-card .value{font-size:2rem;font-weight:700;margin:.2rem 0;}
    .section-divider{border-top:2px solid #e0e0e0;margin:1.5rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# Helper: weather card renderer
# ──────────────────────────────────────────────
def render_weather_card(label: str, value: str, unit: str = "") -> str:
    return (
        f'<div class="weather-card">'
        f'<h3>{label}</h3>'
        f'<div class="value">{value}<span style="font-size:.9rem;opacity:.8"> {unit}</span></div>'
        f"</div>"
    )


# ──────────────────────────────────────────────
# Main app class
# ──────────────────────────────────────────────
class ForecastingApp:
    def __init__(self):
        self.weather_api = WeatherAPI()

    # ── Navigation ────────────────────────────
    def sidebar_navigation(self) -> str:
        st.sidebar.title("🚀 Navigation")
        st.sidebar.markdown("---")
        page = st.sidebar.selectbox(
            "Select Page",
            ["🏠 Home", "📁 Upload Data", "📊 View Data",
             "🌤️ Weather", "🤖 Train Model", "📈 Forecast", "📦 Inventory"],
        )
        st.sidebar.markdown("---")

        # ── User info + logout ─────────────────
        if is_authenticated():
            user_info = st.session_state.get("user_info", {})
            name = user_info.get("name", "User") if user_info else "User"
            st.sidebar.markdown(f"👤 **{name}**")
            st.sidebar.caption(user_info.get("email", "") if user_info else "")
            if st.sidebar.button("🚪 Logout", use_container_width=True):
                logout()
                st.rerun()
            st.sidebar.markdown("---")

        st.sidebar.info("**Built for FMCG Distribution**")
        st.sidebar.caption("Prophet + LightGBM Hybrid Model")
        return page

    # ── Home ──────────────────────────────────
    def home_page(self):
        st.markdown(
            '<h1 class="main-header">📊 Demand Forecasting & Inventory Optimization</h1>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            ### AI-Powered FMCG Distribution Solution
            | Feature | Description |
            |---------|-------------|
            | 🔮 **Prophet** | Captures seasonal trends and holidays |
            | ⚡ **LightGBM** | Leverages engineered lag & rolling features |
            | 🤝 **Hybrid** | Combines both models for optimal accuracy |
            | 🌤️ **Weather** | Real-time OpenWeather API integration |
            | 📦 **Inventory** | Smart reorder-point recommendations |
            """
        )
        col1, col2, col3, col4 = st.columns(4)
        df = st.session_state.df
        if df is not None:
            col1.metric("Total Rows", f"{len(df):,}")
            col2.metric("Total Sales", f"{df['sales'].sum():,.0f}")
            col3.metric("Avg Daily", f"{df['sales'].mean():.1f}")
            col4.metric("Model Trained", "✅ Yes" if st.session_state.trained else "❌ No")
        else:
            col1.metric("Total Sales", "–")
            col2.metric("Avg Daily Demand", "–")
            col3.metric("Forecast Accuracy", "–")
            col4.metric("Stockouts", "–")
        st.info("👆 Start by uploading your sales CSV or loading the sample dataset!")

    # ── Upload ────────────────────────────────
    def upload_data_page(self):
        st.header("📁 Upload Sales Data")
        st.markdown(
            "Upload a CSV with at least **`date`** and **`sales`** columns. "
            "Optional columns: `product`, `inventory`."
        )

        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Accepted columns: date (YYYY-MM-DD), sales, product, inventory",
        )

        if uploaded_file is not None:
            with st.spinner("Reading and validating file…"):
                df, message = load_data_from_upload(uploaded_file)

            if df is None:
                st.error(f"❌ Upload failed:\n\n{message}")
                st.stop()
            else:
                st.session_state.df = df
                # Reset downstream state whenever new data is uploaded
                st.session_state.processed_df = None
                st.session_state.feature_info = None
                st.session_state.forecaster = None
                st.session_state.trained = False

                st.success(f"✅ {message}")
                st.subheader("Dataset Preview")
                st.dataframe(df.head(20), use_container_width=True)

                col1, col2, col3 = st.columns(3)
                col1.metric("Rows", f"{len(df):,}")
                col2.metric("Columns", df.shape[1])
                col3.metric(
                    "Date Range",
                    f"{df['date'].min().date()} → {df['date'].max().date()}",
                )

        st.markdown("---")
        if st.button("🧪 Load Sample Dataset Instead"):
            df = create_sample_data()
            st.session_state.df = df
            st.session_state.processed_df = None
            st.session_state.feature_info = None
            st.session_state.forecaster = None
            st.session_state.trained = False
            st.success("✅ Sample dataset loaded (366 rows, FMCG simulation).")
            st.dataframe(df.head(10), use_container_width=True)

    # ── View Data ─────────────────────────────
    def view_data_page(self):
        st.header("📊 Data Explorer")

        df = st.session_state.df
        if df is None:
            st.warning("👈 Please upload data first!")
            return

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", f"{len(df):,}")
        col2.metric("Total Sales", f"{df['sales'].sum():,.0f}")
        col3.metric("Avg Daily Sales", f"{df['sales'].mean():.1f}")
        col4.metric("Std Dev", f"{df['sales'].std():.1f}")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Dataset table with search/filter
        st.subheader("Raw Dataset")
        search_col = st.text_input("🔍 Filter rows (search any value)", "")
        if search_col:
            mask = df.apply(
                lambda row: row.astype(str).str.contains(search_col, case=False).any(), axis=1
            )
            filtered = df[mask]
            st.caption(f"Showing {len(filtered):,} matching rows")
            st.dataframe(filtered, use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True)

        # Descriptive statistics
        st.subheader("📈 Descriptive Statistics")
        numeric_df = df.select_dtypes(include="number")
        if not numeric_df.empty:
            st.dataframe(numeric_df.describe().round(2), use_container_width=True)
        else:
            st.info("No numeric columns to summarise.")

        # Sales trend chart
        st.subheader("📉 Sales Over Time")
        try:
            fig = px.line(
                df, x="date", y="sales",
                title="Daily Sales Trend",
                labels={"sales": "Sales (units)", "date": "Date"},
            )
            fig.update_traces(line_color="#1f77b4")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Chart error: {e}")

        # Optional: inventory chart
        if "inventory" in df.columns:
            st.subheader("📦 Inventory Levels Over Time")
            try:
                fig2 = px.area(
                    df, x="date", y="inventory",
                    title="Inventory Trend",
                    labels={"inventory": "Inventory (units)", "date": "Date"},
                )
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.error(f"Inventory chart error: {e}")

    # ── Weather ───────────────────────────────
    def weather_page(self):
        st.header("🌤️ Live Weather Data")
        st.markdown(
            "Enter a city name to fetch current weather from **OpenWeatherMap**. "
            "Weather context helps explain demand spikes."
        )

        # City search input
        city_input = st.text_input(
            "🔍 Enter City Name",
            value=st.session_state.selected_city,
            placeholder="e.g. Coimbatore, Mumbai, Delhi",
        )

        fetch_btn = st.button("Fetch Weather", type="primary")

        if fetch_btn:
            if not city_input.strip():
                st.warning("Please enter a city name.")
                return

            st.session_state.selected_city = city_input.strip()

            with st.spinner(f"Fetching weather for **{city_input}**…"):
                result = self.weather_api.get_current_weather(city_input.strip())

            if "error" in result:
                st.error(f"❌ {result['error']}")
                # Contextual help
                if "key" in result["error"].lower():
                    st.info(
                        "**How to fix:** Create a `.env` file in the project root and add:\n"
                        "```\nOPENWEATHER_API_KEY=your_key_here\n```\n"
                        "Get a free key at https://openweathermap.org/api"
                    )
            else:
                st.session_state.weather_data = result
                city_label = f"{result['city']}, {result['country']}"
                st.success(f"✅ Weather fetched for **{city_label}**")

                # Display weather cards
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(
                        render_weather_card("🌡️ Temperature", str(result["temperature"]), "°C"),
                        unsafe_allow_html=True,
                    )
                with col2:
                    st.markdown(
                        render_weather_card("💧 Humidity", str(result["humidity"]), "%"),
                        unsafe_allow_html=True,
                    )
                with col3:
                    st.markdown(
                        render_weather_card("🌥️ Condition", result["condition"]),
                        unsafe_allow_html=True,
                    )
                with col4:
                    st.markdown(
                        render_weather_card("💨 Wind Speed", str(result["wind_speed"]), "m/s"),
                        unsafe_allow_html=True,
                    )

                st.markdown(f"**Feels like:** {result['feels_like']} °C &nbsp;|&nbsp; "
                            f"**Visibility:** {result['visibility']} km")

                # Icon from OpenWeatherMap
                icon_url = f"https://openweathermap.org/img/wn/{result['icon']}@2x.png"
                st.image(icon_url, width=80)

        elif st.session_state.weather_data:
            # Re-display previously fetched data
            w = st.session_state.weather_data
            city_label = f"{w['city']}, {w['country']}"
            st.info(f"Showing cached weather for **{city_label}**. Click 'Fetch Weather' to refresh.")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(render_weather_card("🌡️ Temperature", str(w["temperature"]), "°C"), unsafe_allow_html=True)
            with col2:
                st.markdown(render_weather_card("💧 Humidity", str(w["humidity"]), "%"), unsafe_allow_html=True)
            with col3:
                st.markdown(render_weather_card("🌥️ Condition", w["condition"]), unsafe_allow_html=True)
            with col4:
                st.markdown(render_weather_card("💨 Wind Speed", str(w["wind_speed"]), "m/s"), unsafe_allow_html=True)

        # API key status
        with st.expander("⚙️ API Key Status"):
            key_set = bool(os.getenv("OPENWEATHER_API_KEY", ""))
            if key_set:
                st.success("✅ OPENWEATHER_API_KEY is configured.")
            else:
                st.error("❌ OPENWEATHER_API_KEY is NOT set.")
                st.code("OPENWEATHER_API_KEY=your_key_here", language="bash")
                st.markdown("Create a `.env` file in the project root with the above, then restart the app.")

    # ── Train Model ───────────────────────────
    def train_model_page(self):
        st.header("🤖 Model Training")

        df = st.session_state.df
        if df is None:
            st.warning("👈 Please upload data first!")
            return

        city = st.session_state.selected_city or "Coimbatore"
        st.info(f"Training will use **historical weather simulation** for {city}.")

        if st.button("🚀 Train All Models", type="primary", use_container_width=True):
            progress = st.progress(0, text="Starting…")
            status_box = st.empty()

            try:
                # Step 1: fetch historical weather
                progress.progress(15, text="Fetching historical weather data…")
                weather_df = self.weather_api.get_historical_weather(
                    df["date"].min().strftime("%Y-%m-%d"),
                    df["date"].max().strftime("%Y-%m-%d"),
                    city=city,
                )

                # Step 2: preprocess
                progress.progress(35, text="Preprocessing data…")
                processed_df, feature_info = preprocess_data(df, weather_df)
                st.session_state.processed_df = processed_df
                st.session_state.feature_info = feature_info

                # Step 3: train
                progress.progress(55, text="Training Prophet model…")
                forecaster = DemandForecaster()
                results = forecaster.train_all(processed_df, feature_info)
                st.session_state.forecaster = forecaster
                st.session_state.trained = True

                progress.progress(100, text="Done!")
                status_box.success("✅ Training complete!")

                # Show results
                for res in results:
                    if res["status"] == "success":
                        extra = f", RMSE: {res['rmse']}" if "rmse" in res else ""
                        st.metric(res["model"], f"MAE: {res['mae']}{extra}")
                    else:
                        st.error(f"❌ {res['model']} failed: {res['error']}")

            except ValueError as e:
                st.error(f"Validation error: {e}")
            except Exception as e:
                st.error(f"Training failed unexpectedly: {e}")
                import traceback
                with st.expander("Stack trace"):
                    st.code(traceback.format_exc())

    # ── Forecast ──────────────────────────────
    def forecast_page(self):
        st.header("📈 Demand Forecast")

        if not st.session_state.trained:
            st.warning("⚠️ Please train the model first (🤖 Train Model page).")
            return

        forecaster = st.session_state.forecaster
        processed_df = st.session_state.processed_df
        feature_info = st.session_state.feature_info

        if forecaster is None or processed_df is None or feature_info is None:
            st.error("❌ Session state incomplete. Please retrain the model.")
            return

        periods = st.slider("Forecast Horizon (days)", 7, 90, 30)

        if st.button("🔮 Generate Forecast", type="primary"):
            with st.spinner("Generating predictions…"):
                try:
                    forecasts = forecaster.forecast(processed_df, periods, feature_info)
                except Exception as e:
                    st.error(f"Forecast error: {e}")
                    return

            # ── Build chart ──────────────────
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Demand Forecast", "LightGBM vs Prophet"),
                vertical_spacing=0.12,
                row_heights=[0.7, 0.3],
            )

            hist_df = processed_df.tail(90)

            # Historical
            fig.add_trace(
                go.Scatter(x=hist_df["date"], y=hist_df["sales"],
                           mode="lines+markers", name="Historical Sales",
                           line=dict(color="#1f77b4")),
                row=1, col=1,
            )

            # Prophet forecast + CI
            prophet_fc = forecasts["prophet"]
            fig.add_trace(
                go.Scatter(x=prophet_fc["ds"], y=prophet_fc["yhat"],
                           mode="lines", name="Prophet Forecast",
                           line=dict(color="#ff7f0e", width=2)),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(x=prophet_fc["ds"], y=prophet_fc["yhat_upper"],
                           mode="lines", line=dict(width=0), showlegend=False),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(x=prophet_fc["ds"], y=prophet_fc["yhat_lower"],
                           fill="tonexty", fillcolor="rgba(255,127,14,0.15)",
                           line=dict(width=0), name="Confidence Interval"),
                row=1, col=1,
            )

            # LightGBM forecast
            if "lgbm" in forecasts:
                future_dates = pd.date_range(
                    start=processed_df["date"].max() + timedelta(days=1),
                    periods=periods,
                    freq="D",
                )
                fig.add_trace(
                    go.Scatter(x=future_dates, y=forecasts["lgbm"],
                               mode="lines+markers", name="LightGBM Forecast",
                               line=dict(color="#2ca02c", dash="dash")),
                    row=1, col=1,
                )
                # Comparison chart (row 2)
                fig.add_trace(
                    go.Bar(x=future_dates, y=forecasts["lgbm"] - prophet_fc["yhat"].values,
                           name="LightGBM − Prophet diff",
                           marker_color="rgba(44,160,44,0.6)"),
                    row=2, col=1,
                )

            fig.update_layout(height=650, title="AI-Powered Demand Forecasting",
                              hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            # Summary metric
            avg_fc = prophet_fc["yhat"].mean()
            st.metric("Avg Forecasted Demand (Prophet)", f"{avg_fc:.0f} units/day")

            # Detailed table
            st.subheader("Detailed Forecast Table")
            table = prophet_fc.copy()
            table.columns = ["Date", "Forecast", "Lower", "Upper"]
            table["Date"] = table["Date"].dt.date
            table = table.round(1)
            st.dataframe(table, use_container_width=True)

    # ── Inventory ─────────────────────────────
    def inventory_page(self):
        st.header("📦 Inventory Optimization")

        processed_df = st.session_state.processed_df
        if processed_df is None:
            # Fall back to raw df if preprocessing hasn't run
            df_source = st.session_state.df
            if df_source is None:
                st.warning("👈 Upload data and train models first!")
                return
        else:
            df_source = processed_df

        col1, col2 = st.columns(2)
        with col1:
            lead_time = st.number_input("Lead Time (days)", min_value=1, max_value=60, value=7)
        with col2:
            safety_stock = st.number_input("Safety Stock (units)", min_value=0, max_value=1000, value=50)

        if st.button("📐 Calculate Reorder Point", type="primary"):
            try:
                avg_demand = float(df_source["sales"].tail(30).mean())
                metrics = calculate_inventory_metrics(avg_demand, lead_time, safety_stock)

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Avg Daily Demand", f"{metrics['average_daily_demand']} units")
                col2.metric("Lead Time", f"{metrics['lead_time']} days")
                col3.metric("Safety Stock", f"{metrics['safety_stock']} units")
                col4.metric("🔔 Reorder Point", f"{metrics['reorder_point']} units")

                st.success(metrics["recommendation"])
                st.info(f"💡 Suggested Order Qty: **{metrics['suggested_order_qty']:.0f} units**")

                # Distribution chart
                fig = px.histogram(
                    df_source.tail(90), x="sales", nbins=25,
                    title="Recent 90-Day Demand Distribution",
                    labels={"sales": "Daily Sales (units)"},
                    color_discrete_sequence=["#1f77b4"],
                )
                fig.add_vline(x=metrics["reorder_point"] / lead_time,
                              line_dash="dash", line_color="red",
                              annotation_text="Avg Demand Threshold")
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Calculation error: {e}")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
def main():
    # ── Auth gate: show login/register if not authenticated ──
    if not require_auth():
        return

    app = ForecastingApp()
    page = app.sidebar_navigation()

    if page == "🏠 Home":
        app.home_page()
    elif page == "📁 Upload Data":
        app.upload_data_page()
    elif page == "📊 View Data":
        app.view_data_page()
    elif page == "🌤️ Weather":
        app.weather_page()
    elif page == "🤖 Train Model":
        app.train_model_page()
    elif page == "📈 Forecast":
        app.forecast_page()
    elif page == "📦 Inventory":
        app.inventory_page()


if __name__ == "__main__":
    main()
