"""
Streamlit Dashboard for Demand Forecasting & Inventory Optimization
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import numpy as np
from datetime import datetime, timedelta

# Import project modules
from utils import load_data, preprocess_data, calculate_inventory_metrics, create_sample_data
from model import DemandForecaster
from api import WeatherAPI

# Initialize session state
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'feature_info' not in st.session_state:
    st.session_state.feature_info = None
if 'forecaster' not in st.session_state:
    st.session_state.forecaster = None
# Page configuration
st.set_page_config(
    page_title="Demand Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

class ForecastingApp:
    def __init__(self):
        self.df = st.session_state.df
        self.processed_df = st.session_state.processed_df
        self.forecaster = st.session_state.forecaster
        self.weather_api = WeatherAPI()
        self.feature_info = st.session_state.feature_info
    def sidebar_navigation(self):
        """Create professional sidebar navigation"""
        st.sidebar.title("🚀 Navigation")
        st.sidebar.markdown("---")
        
        page = st.sidebar.selectbox(
            "Select Page",
            ["🏠 Home", "📁 Upload Data", "📊 View Data", "🤖 Train Model", 
             "📈 Forecast", "📦 Inventory"]
        )
        
        st.sidebar.markdown("---")
        st.sidebar.info("**Built for FMCG Distribution**")
        st.sidebar.caption("Prophet + LightGBM Hybrid Model")
        
        return page
    def home_page(self):
        """Home page with KPIs and project info"""
        st.markdown('<h1 class="main-header">📊 Demand Forecasting & Inventory Optimization</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        ### AI-Powered FMCG Distribution Solution
        - **Prophet**: Captures seasonal trends and holidays
        - **LightGBM**: Leverages engineered features and lags  
        - **Hybrid Model**: Combines both for optimal accuracy
        - **Weather Integration**: External factors affecting demand
        - **Inventory Optimization**: Smart reorder recommendations
        """)
        
        # Create sample KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Sales", "₹4.2L", "12%")
        with col2:
            st.metric("Avg Daily Demand", "165 units", "3%")
        with col3:
            st.metric("Forecast Accuracy", "92.5%", "2%")
        with col4:
            st.metric("Stockouts", "2 days", "-50%")
        
        st.markdown("---")
        st.info("👆 Upload your sales data (CSV) to get started!")
    def upload_data_page(self):
        """Upload and store CSV data"""
        st.header("📁 Upload Sales Data")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose CSV file (date, sales columns required)",
            type="csv",
            help="Expected format: date,sales,product_id,location"
        )
        
        if uploaded_file is not None:
            # Save uploaded file
            save_path = "data/uploaded_sales.csv"
            os.makedirs("data", exist_ok=True)
            
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load and display preview
            @st.cache_data
            def load_uploaded():
                return load_data(save_path)
            
            self.df = load_uploaded()
            st.session_state.df = self.df
            st.success(f"✅ Data uploaded! Shape: {self.df.shape}")
            
            # Show preview
            st.subheader("Data Preview")
            st.dataframe(self.df.head(10), use_container_width=True)
            
        elif st.button("🧪 Use Sample Data"):
            self.df = create_sample_data()
            st.session_state.df = self.df
            st.success("✅ Sample data loaded!")

    def view_data_page(self):
        """Display processed data and statistics"""
        st.header("📊 Data Explorer")
        
        if st.session_state.df is not None:
            df = st.session_state.df.copy()
            
            # Data summary
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.metric("Dataset Size", f"{len(df):,}")
                st.metric("Date Range", f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
            
            with col2:
                st.metric("Total Sales", f"₹{df['sales'].sum():,.0f}")
                st.metric("Avg Daily", f"{df['sales'].mean():.0f} units")
            
            # Raw data table
            st.subheader("Raw Dataset")
            st.dataframe(df, use_container_width=True)
            
            # Statistics
            st.subheader("📈 Data Statistics")
            st.dataframe(df.describe(), use_container_width=True)
            
        else:
            st.warning("👈 Please upload data first!")

    def train_model_page(self):
        """Train ML models"""
        st.header("🤖 Model Training")
        
        if st.session_state.df is not None:
            df = st.session_state.df.copy()
            
            if st.button("🚀 Train All Models", type="primary", use_container_width=True):
                with st.spinner("Training Prophet + LightGBM models..."):
                    # Initialize forecaster if not exists
                    if st.session_state.forecaster is None:
                        st.session_state.forecaster = DemandForecaster()
                    
                    # Fetch weather data
                    weather_df = self.weather_api.get_historical_weather(
                        df['date'].min().strftime('%Y-%m-%d'),
                        df['date'].max().strftime('%Y-%m-%d')
                    )
                    
                    # Preprocess
                    self.processed_df, self.feature_info = preprocess_data(df, weather_df)
                    st.session_state.processed_df = self.processed_df
                    st.session_state.feature_info = self.feature_info
                    
                    # Train models
                    training_results = st.session_state.forecaster.train_all(self.processed_df, self.feature_info)
                    
                    # Update session state
                    st.session_state.trained = True
                    
                    # Display results
                    st.success("✅ Training completed!")
                    for result in training_results:
                        if result['status'] == 'success':
                            st.metric(result['model'], f"MAE: {result['mae']}")
                        else:
                            st.error(f"{result['model']}: {result['error']}")
        else:
            st.warning("👈 Please upload data first!")

    def forecast_page(self):
        """Generate and visualize forecasts"""
        st.header("📈 Demand Forecast")
        
        # Check if model is trained
        if not st.session_state.get('trained', False):
            st.warning("⚠️ Please train the model first")
            return
        
        # Check if required session state exists
        if (st.session_state.forecaster is None or 
            st.session_state.processed_df is None or 
            st.session_state.feature_info is None):
            st.error("❌ Session state corrupted. Please retrain the model.")
            return
        
        # Load from session
        forecaster = st.session_state.forecaster
        processed_df = st.session_state.processed_df
        feature_info = st.session_state.feature_info

        # Forecast controls
        periods = st.slider("Forecast Horizon (days)", 7, 90, 30)

        if st.button("🔮 Generate Forecast", type="primary"):
            with st.spinner("Generating predictions..."):
                forecasts = forecaster.forecast(processed_df, periods, feature_info)

                # Create plot
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Demand Forecast', 'Confidence Intervals'),
                    vertical_spacing=0.1,
                    row_heights=[0.7, 0.3]
                )

                # Historical data
                hist_df = processed_df.tail(60)

                fig.add_trace(
                    go.Scatter(
                        x=hist_df['date'],
                        y=hist_df['sales'],
                        mode='lines+markers',
                        name='Historical Sales'
                    ),
                    row=1, col=1
                )

                # Prophet forecast
                prophet_fc = forecasts['prophet']

                fig.add_trace(
                    go.Scatter(
                        x=prophet_fc['ds'],
                        y=prophet_fc['yhat'],
                        mode='lines',
                        name='Prophet Forecast'
                    ),
                    row=1, col=1
                )

                # Confidence interval
                fig.add_trace(
                    go.Scatter(
                        x=prophet_fc['ds'],
                        y=prophet_fc['yhat_upper'],
                        line=dict(width=0),
                        showlegend=False
                    ),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=prophet_fc['ds'],
                        y=prophet_fc['yhat_lower'],
                        fill='tonexty',
                        name='Confidence Interval'
                    ),
                    row=1, col=1
                )

                # LightGBM forecast
                if 'lgbm' in forecasts:
                    future_dates = pd.date_range(
                        start=processed_df['date'].max() + timedelta(days=1),
                        periods=periods,
                        freq='D'
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=future_dates,
                            y=forecasts['lgbm'],
                            mode='lines+markers',
                            name='LightGBM Forecast'
                        ),
                        row=1, col=1
                    )

                # Metrics
                avg_forecast = prophet_fc['yhat'].tail(periods).mean()
                st.metric("Avg Forecasted Demand", f"{avg_forecast:.0f} units/day")

                # Layout
                fig.update_layout(
                    height=600,
                    title="AI-Powered Demand Forecasting",
                    hovermode='x unified'
                )

                st.plotly_chart(fig, use_container_width=True)

                # Table
                st.subheader("Detailed Predictions")

                forecast_table = prophet_fc[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
                forecast_table.columns = ['Date', 'Predicted', 'Lower', 'Upper']

                st.dataframe(forecast_table, use_container_width=True)
    
    def inventory_page(self):
        """Inventory optimization calculator"""
        st.header("📦 Inventory Optimization")
        
        if st.session_state.processed_df is not None:
            df = st.session_state.processed_df
            
            col1, col2 = st.columns(2)
            
            with col1:
                lead_time = st.number_input("Lead Time (days)", min_value=1, max_value=30, value=7)
            
            with col2:
                safety_stock = st.number_input("Safety Stock (units)", min_value=0, max_value=500, value=50)
            
            if st.button("Calculate Reorder Point", type="primary"):
                avg_demand = df['sales'].tail(30).mean()
                metrics = calculate_inventory_metrics(avg_demand, lead_time, safety_stock)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Avg Daily Demand", f"{metrics['average_daily_demand']} units")
                with col2:
                    st.metric("Lead Time", f"{metrics['lead_time']} days")
                with col3:
                    st.metric("Safety Stock", f"{metrics['safety_stock']} units")
                with col4:
                    st.metric("🔔 Reorder Point", f"{metrics['reorder_point']} units")
                
                st.success(metrics['recommendation'])
                
                # Demand distribution chart
                fig = px.histogram(df.tail(90), x='sales', nbins=20,
                                 title="Recent Demand Distribution",
                                 labels={'sales': 'Daily Sales (Units)'})
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("👈 Upload data and train models first!")

    def main():
    app = ForecastingApp()
    
    # Sidebar navigation
    page = app.sidebar_navigation()
    
    # Render selected page
    if page == "🏠 Home":
        app.home_page()
    elif page == "📁 Upload Data":
        app.upload_data_page()
    elif page == "📊 View Data":
        app.view_data_page()
    elif page == "🤖 Train Model":
        app.train_model_page()
    elif page == "📈 Forecast":
        app.forecast_page()
    elif page == "📦 Inventory":
        app.inventory_page()

if __name__ == "__main__":
    main()