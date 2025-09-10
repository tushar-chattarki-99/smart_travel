import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Smart Travel Expense Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 10px;
    }
    
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .expense-input {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .alert-overspend {
        background: #fff5f5;
        border: 1px solid #fed7d7;
        border-radius: 8px;
        padding: 1rem;
        color: #c53030;
    }
    
    .alert-under-budget {
        background: #f0fff4;
        border: 1px solid #c6f6d5;
        border-radius: 8px;
        padding: 1rem;
        color: #22543d;
    }
</style>
""", unsafe_allow_html=True)

class TravelExpensePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def create_sample_dataset(self):
        """Create a comprehensive sample dataset of travel expenses"""
        np.random.seed(42)
        
        destinations = [
            'Paris', 'Tokyo', 'New York', 'London', 'Dubai', 'Bangkok', 'Sydney',
            'Rome', 'Barcelona', 'Amsterdam', 'Berlin', 'Prague', 'Vienna',
            'Mumbai', 'Delhi', 'Goa', 'Bangalore', 'Jaipur', 'Kerala', 'Agra',
            'Bali', 'Singapore', 'Kuala Lumpur', 'Seoul', 'Hong Kong'
        ]
        
        trip_types = ['Budget', 'Standard', 'Luxury']
        
        # Base costs per day per person for different destinations (USD)
        destination_costs = {
            'Paris': {'Budget': 80, 'Standard': 150, 'Luxury': 300},
            'Tokyo': {'Budget': 90, 'Standard': 170, 'Luxury': 350},
            'New York': {'Budget': 100, 'Standard': 200, 'Luxury': 400},
            'London': {'Budget': 85, 'Standard': 160, 'Luxury': 320},
            'Dubai': {'Budget': 70, 'Standard': 140, 'Luxury': 280},
            'Bangkok': {'Budget': 40, 'Standard': 80, 'Luxury': 160},
            'Sydney': {'Budget': 95, 'Standard': 180, 'Luxury': 360},
            'Rome': {'Budget': 75, 'Standard': 140, 'Luxury': 280},
            'Barcelona': {'Budget': 70, 'Standard': 130, 'Luxury': 260},
            'Amsterdam': {'Budget': 80, 'Standard': 150, 'Luxury': 300},
            'Berlin': {'Budget': 65, 'Standard': 120, 'Luxury': 240},
            'Prague': {'Budget': 50, 'Standard': 90, 'Luxury': 180},
            'Vienna': {'Budget': 70, 'Standard': 130, 'Luxury': 260},
            'Mumbai': {'Budget': 30, 'Standard': 60, 'Luxury': 120},
            'Delhi': {'Budget': 35, 'Standard': 70, 'Luxury': 140},
            'Goa': {'Budget': 40, 'Standard': 80, 'Luxury': 160},
            'Bangalore': {'Budget': 35, 'Standard': 70, 'Luxury': 140},
            'Jaipur': {'Budget': 30, 'Standard': 60, 'Luxury': 120},
            'Kerala': {'Budget': 40, 'Standard': 80, 'Luxury': 160},
            'Agra': {'Budget': 25, 'Standard': 50, 'Luxury': 100},
            'Bali': {'Budget': 45, 'Standard': 90, 'Luxury': 180},
            'Singapore': {'Budget': 80, 'Standard': 150, 'Luxury': 300},
            'Kuala Lumpur': {'Budget': 35, 'Standard': 70, 'Luxury': 140},
            'Seoul': {'Budget': 70, 'Standard': 130, 'Luxury': 260},
            'Hong Kong': {'Budget': 85, 'Standard': 160, 'Luxury': 320}
        }
        
        data = []
        
        for _ in range(1000):  # Generate 1000 sample trips
            destination = np.random.choice(destinations)
            trip_type = np.random.choice(trip_types)
            days = np.random.randint(3, 21)  # 3-20 days
            travelers = np.random.randint(1, 7)  # 1-6 travelers
            
            # Get base cost per day per person
            base_cost = destination_costs[destination][trip_type]
            
            # Add some randomness (±20%)
            variation = np.random.normal(1, 0.15)
            daily_cost_per_person = max(base_cost * variation, base_cost * 0.5)
            
            total_cost = daily_cost_per_person * days * travelers
            
            # Break down into categories (percentages)
            transport_pct = np.random.uniform(0.15, 0.35)  # 15-35%
            accommodation_pct = np.random.uniform(0.25, 0.45)  # 25-45%
            food_pct = np.random.uniform(0.15, 0.30)  # 15-30%
            activities_pct = 1 - transport_pct - accommodation_pct - food_pct
            
            # Ensure activities percentage is positive
            if activities_pct < 0.05:
                activities_pct = 0.05
                # Normalize other percentages
                total_other = transport_pct + accommodation_pct + food_pct
                transport_pct = transport_pct * (0.95 / total_other)
                accommodation_pct = accommodation_pct * (0.95 / total_other)
                food_pct = food_pct * (0.95 / total_other)
            
            data.append({
                'destination': destination,
                'days': days,
                'trip_type': trip_type,
                'travelers': travelers,
                'total_cost': round(total_cost, 2),
                'transport_cost': round(total_cost * transport_pct, 2),
                'accommodation_cost': round(total_cost * accommodation_pct, 2),
                'food_cost': round(total_cost * food_pct, 2),
                'activities_cost': round(total_cost * activities_pct, 2)
            })
        
        return pd.DataFrame(data)
    
    def train_model(self, df):
        """Train the expense prediction model"""
        # Prepare features
        features = ['destination', 'days', 'trip_type', 'travelers']
        X = df[features].copy()
        
        # Encode categorical variables
        for col in ['destination', 'trip_type']:
            self.label_encoders[col] = LabelEncoder()
            X[col] = self.label_encoders[col].fit_transform(X[col])
        
        # Scale numerical features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train models for different cost categories
        self.models = {}
        cost_categories = ['total_cost', 'transport_cost', 'accommodation_cost', 
                          'food_cost', 'activities_cost']
        
        for category in cost_categories:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_scaled, df[category])
            self.models[category] = model
        
        self.is_trained = True
        return True
    
    def predict_expenses(self, destination, days, trip_type, travelers):
        """Predict expenses for given trip parameters"""
        if not self.is_trained:
            return None
        
        # Prepare input data
        input_data = pd.DataFrame({
            'destination': [destination],
            'days': [days],
            'trip_type': [trip_type],
            'travelers': [travelers]
        })
        
        # Encode categorical variables
        for col in ['destination', 'trip_type']:
            if col in self.label_encoders:
                try:
                    input_data[col] = self.label_encoders[col].transform(input_data[col])
                except ValueError:
                    # Handle unseen categories - use most similar category
                    input_data[col] = 0  # Default to first category
        
        # Scale features
        input_scaled = self.scaler.transform(input_data)
        
        # Predict expenses
        predictions = {}
        for category, model in self.models.items():
            pred = model.predict(input_scaled)[0]
            predictions[category] = max(0, round(pred, 2))  # Ensure non-negative
        
        return predictions

def load_or_create_dataset():
    """Load existing dataset or create a new one"""
    dataset_file = 'travel_expenses.csv'
    
    if os.path.exists(dataset_file):
        df = pd.read_csv(dataset_file)
    else:
        predictor = TravelExpensePredictor()
        df = predictor.create_sample_dataset()
        df.to_csv(dataset_file, index=False)
        st.success(f"Created sample dataset with {len(df)} travel records!")
    
    return df

def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'actual_expenses' not in st.session_state:
        st.session_state.actual_expenses = {}
    if 'expense_history' not in st.session_state:
        st.session_state.expense_history = []

def create_budget_vs_actual_chart(predictions, actual_expenses):
    """Create pie chart comparing budget vs actual expenses"""
    if not predictions or not actual_expenses:
        return None
    
    categories = ['Transport', 'Accommodation', 'Food', 'Activities']
    budget_values = [
        predictions['transport_cost'],
        predictions['accommodation_cost'], 
        predictions['food_cost'],
        predictions['activities_cost']
    ]
    
    actual_values = [
        actual_expenses.get('transport', 0),
        actual_expenses.get('accommodation', 0),
        actual_expenses.get('food', 0),
        actual_expenses.get('activities', 0)
    ]
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type':'domain'}, {'type':'domain'}]],
        subplot_titles=['Estimated Budget', 'Actual Spending']
    )
    
    # Budget pie chart
    fig.add_trace(go.Pie(
        labels=categories,
        values=budget_values,
        name="Budget",
        marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    ), 1, 1)
    
    # Actual pie chart
    fig.add_trace(go.Pie(
        labels=categories,
        values=actual_values,
        name="Actual",
        marker_colors=['#FF8E8E', '#6ED5CD', '#67C3D6', '#A8D5BA']
    ), 1, 2)
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        title_text="Budget vs Actual Expenses Breakdown",
        title_x=0.5,
        showlegend=True,
        height=500
    )
    
    return fig

def create_daily_spending_chart(expense_history):
    """Create daily spending line chart"""
    if not expense_history:
        return None
    
    df_history = pd.DataFrame(expense_history)
    df_history['date'] = pd.to_datetime(df_history['date'])
    
    fig = px.line(
        df_history, 
        x='date', 
        y='amount',
        color='category',
        title='Daily Spending Tracker',
        labels={'amount': 'Amount (USD)', 'date': 'Date'}
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Amount (USD)",
        hovermode='x unified'
    )
    
    return fig

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>✈️ Smart Travel Expense Planner</h1>
        <p>AI-powered trip cost prediction and expense tracking</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Load dataset and train model
    @st.cache_data
    def load_data_and_train():
        df = load_or_create_dataset()
        predictor = TravelExpensePredictor()
        predictor.train_model(df)
        return df, predictor
    
    df, predictor = load_data_and_train()
    
    # Sidebar for input parameters
    st.sidebar.header("🎯 Trip Details")
    
    # Get unique destinations from dataset
    destinations = sorted(df['destination'].unique())
    
    with st.sidebar:
        destination = st.selectbox("📍 Destination", destinations)
        days = st.slider("📅 Number of Days", min_value=1, max_value=30, value=7)
        trip_type = st.selectbox("💰 Trip Type", ["Budget", "Standard", "Luxury"])
        travelers = st.slider("👥 Number of Travelers", min_value=1, max_value=10, value=2)
        
        predict_button = st.button("🔮 Predict Expenses", type="primary")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Prediction section
        if predict_button or st.session_state.predictions:
            if predict_button:
                with st.spinner("🤖 AI is analyzing travel costs..."):
                    predictions = predictor.predict_expenses(destination, days, trip_type, travelers)
                    st.session_state.predictions = predictions
            
            predictions = st.session_state.predictions
            
            if predictions:
                st.subheader("📊 Estimated Trip Costs")
                
                # Display total cost prominently
                st.metric(
                    label="💵 Total Estimated Cost",
                    value=f"${predictions['total_cost']:,.2f}",
                    delta=f"${predictions['total_cost']/travelers:,.2f} per person"
                )
                
                # Cost breakdown
                col1a, col1b, col1c, col1d = st.columns(4)
                
                with col1a:
                    st.metric("🚗 Transport", f"${predictions['transport_cost']:,.2f}")
                
                with col1b:
                    st.metric("🏨 Accommodation", f"${predictions['accommodation_cost']:,.2f}")
                
                with col1c:
                    st.metric("🍽️ Food", f"${predictions['food_cost']:,.2f}")
                
                with col1d:
                    st.metric("🎯 Activities", f"${predictions['activities_cost']:,.2f}")
                
                # Actual expenses input
                st.subheader("💰 Track Actual Expenses")
                st.markdown("Enter your actual expenses to compare with predictions:")
                
                col_t, col_a, col_f, col_act = st.columns(4)
                
                with col_t:
                    transport_actual = st.number_input(
                        "🚗 Transport ($)",
                        min_value=0.0,
                        value=float(st.session_state.actual_expenses.get('transport', 0)),
                        step=10.0,
                        key="transport_input"
                    )
                
                with col_a:
                    accommodation_actual = st.number_input(
                        "🏨 Accommodation ($)",
                        min_value=0.0,
                        value=float(st.session_state.actual_expenses.get('accommodation', 0)),
                        step=10.0,
                        key="accommodation_input"
                    )
                
                with col_f:
                    food_actual = st.number_input(
                        "🍽️ Food ($)",
                        min_value=0.0,
                        value=float(st.session_state.actual_expenses.get('food', 0)),
                        step=5.0,
                        key="food_input"
                    )
                
                with col_act:
                    activities_actual = st.number_input(
                        "🎯 Activities ($)",
                        min_value=0.0,
                        value=float(st.session_state.actual_expenses.get('activities', 0)),
                        step=5.0,
                        key="activities_input"
                    )
                
                # Update actual expenses in session state
                st.session_state.actual_expenses = {
                    'transport': transport_actual,
                    'accommodation': accommodation_actual,
                    'food': food_actual,
                    'activities': activities_actual
                }
                
                total_actual = sum(st.session_state.actual_expenses.values())
                
                if total_actual > 0:
                    # Budget analysis
                    st.subheader("📈 Budget Analysis")
                    
                    budget_difference = predictions['total_cost'] - total_actual
                    remaining_budget = budget_difference
                    
                    col_budget1, col_budget2, col_budget3 = st.columns(3)
                    
                    with col_budget1:
                        st.metric("💵 Total Actual", f"${total_actual:,.2f}")
                    
                    with col_budget2:
                        st.metric(
                            "💰 Budget Status",
                            f"${abs(budget_difference):,.2f}",
                            delta=f"{'Under' if budget_difference >= 0 else 'Over'} budget"
                        )
                    
                    with col_budget3:
                        accuracy = (1 - abs(budget_difference) / predictions['total_cost']) * 100
                        st.metric("🎯 Prediction Accuracy", f"{max(0, accuracy):.1f}%")
                    
                    # Budget alert
                    if budget_difference < 0:
                        st.markdown(f"""
                        <div class="alert-overspend">
                            <strong>⚠️ Over Budget Alert!</strong><br>
                            You've exceeded your budget by ${abs(budget_difference):,.2f}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="alert-under-budget">
                            <strong>✅ Within Budget!</strong><br>
                            You have ${budget_difference:,.2f} remaining in your budget
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Visualizations
                    st.subheader("📊 Expense Visualizations")
                    
                    # Budget vs Actual chart
                    budget_chart = create_budget_vs_actual_chart(predictions, st.session_state.actual_expenses)
                    if budget_chart:
                        st.plotly_chart(budget_chart, use_container_width=True)
                    
                    # Category comparison
                    comparison_data = {
                        'Category': ['Transport', 'Accommodation', 'Food', 'Activities'],
                        'Estimated': [predictions['transport_cost'], predictions['accommodation_cost'], 
                                    predictions['food_cost'], predictions['activities_cost']],
                        'Actual': [transport_actual, accommodation_actual, food_actual, activities_actual]
                    }
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    comparison_df['Difference'] = comparison_df['Actual'] - comparison_df['Estimated']
                    
                    fig_comparison = px.bar(
                        comparison_df,
                        x='Category',
                        y=['Estimated', 'Actual'],
                        title='Estimated vs Actual Expenses by Category',
                        barmode='group',
                        color_discrete_map={'Estimated': '#667eea', 'Actual': '#764ba2'}
                    )
                    
                    st.plotly_chart(fig_comparison, use_container_width=True)
    
    with col2:
        # Dataset insights
        st.subheader("📈 Dataset Insights")
        
        # Average costs by trip type
        avg_costs = df.groupby('trip_type')['total_cost'].mean().round(2)
        st.write("**Average costs by trip type:**")
        for trip_type, cost in avg_costs.items():
            st.write(f"• {trip_type}: ${cost:,.2f}")
        
        # Popular destinations
        top_destinations = df['destination'].value_counts().head(5)
        st.write("**Most popular destinations:**")
        for dest, count in top_destinations.items():
            st.write(f"• {dest}: {count} trips")
        
        # Trip duration insights
        avg_duration = df['days'].mean()
        st.write(f"**Average trip duration:** {avg_duration:.1f} days")
        
        # Cost per day insights
        df['cost_per_day'] = df['total_cost'] / df['days']
        avg_cost_per_day = df.groupby('trip_type')['cost_per_day'].mean().round(2)
        st.write("**Average cost per day:**")
        for trip_type, cost in avg_cost_per_day.items():
            st.write(f"• {trip_type}: ${cost:,.2f}/day")
        
        # Quick tips
        st.subheader("💡 Money-Saving Tips")
        st.markdown("""
        - **Budget trips**: Focus on local transportation and street food
        - **Standard trips**: Balance comfort with cost-effective choices  
        - **Luxury trips**: Book premium services in advance for better deals
        - **Group travel**: Split accommodation and transportation costs
        - **Off-season**: Travel during shoulder seasons for better prices
        """)
        
        # Export functionality
        st.subheader("📥 Export Data")
        if st.session_state.predictions and total_actual > 0:
            export_data = {
                'destination': destination,
                'days': days,
                'trip_type': trip_type,
                'travelers': travelers,
                'estimated_total': predictions['total_cost'],
                'actual_total': total_actual,
                'budget_difference': budget_difference,
                'categories': {
                    'transport': {'estimated': predictions['transport_cost'], 'actual': transport_actual},
                    'accommodation': {'estimated': predictions['accommodation_cost'], 'actual': accommodation_actual},
                    'food': {'estimated': predictions['food_cost'], 'actual': food_actual},
                    'activities': {'estimated': predictions['activities_cost'], 'actual': activities_actual}
                }
            }
            
            export_json = json.dumps(export_data, indent=2)
            st.download_button(
                label="📁 Download Trip Report",
                data=export_json,
                file_name=f"trip_report_{destination}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()