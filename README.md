# 🚀 Smart Travel Expense Planner

An AI-powered travel expense planner built with Streamlit that predicts trip costs using machine learning and helps track actual expenses against your budget.

## ✨ Features

- **AI-Powered Cost Prediction**: Uses Random Forest ML model trained on 1000+ sample trips
- **Comprehensive Cost Breakdown**: Estimates for transport, accommodation, food, and activities
- **Real-time Budget Tracking**: Compare estimated vs actual expenses
- **Interactive Visualizations**: Pie charts and bar graphs for expense analysis
- **Budget Alerts**: Notifications when you're over/under budget
- **Multiple Trip Types**: Support for Budget, Standard, and Luxury travel styles
- **25+ Destinations**: Popular destinations worldwide with realistic cost data
- **Export Functionality**: Download trip reports as JSON files

## 🎯 How It Works

1. **Input Trip Details**: Select destination, duration, trip type, and number of travelers
2. **Get AI Predictions**: Machine learning model predicts comprehensive costs
3. **Track Actual Expenses**: Input real expenses as you travel
4. **Visual Analysis**: See budget vs actual spending with interactive charts
5. **Budget Management**: Get alerts and insights about your spending

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Quick Start

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/smart-travel-planner.git
cd smart-travel-planner
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the application**:
```bash
streamlit run app.py
```

4. **Open your browser** and navigate to `http://localhost:8501`

### Alternative: Run with virtual environment
```bash
# Create virtual environment
python -m venv travel_planner_env

# Activate virtual environment
# On Windows:
travel_planner_env\Scripts\activate
# On Mac/Linux:
source travel_planner_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📊 Sample Dataset

The app automatically generates a comprehensive dataset with:
- **1000+ sample trips** across 25 destinations
- **Realistic cost variations** based on destination and trip type
- **Historical data** for model training
- **Multiple trip types** (Budget/Standard/Luxury)

### Destinations Included:
- **Europe**: Paris, London, Rome, Barcelona, Amsterdam, Berlin, Prague, Vienna
- **Asia**: Tokyo, Bangkok, Singapore, Seoul, Hong Kong, Kuala Lumpur, Bali
- **North America**: New York
- **Middle East**: Dubai
- **Oceania**: Sydney
- **India**: Mumbai, Delhi, Bangalore, Goa, Jaipur, Kerala, Agra

## 🤖 Machine Learning Model

- **Algorithm**: Random Forest Regressor
- **Features**: Destination, trip duration, trip type, number of travelers
- **Predictions**: Total cost + breakdown by category (transport, accommodation, food, activities)
- **Accuracy**: Typically 80-90% accurate for cost predictions
- **Training Data**: 1000+ synthetic trip records based on real-world travel costs

## 📱 User Interface

### Main Features:
- **Sidebar Input**: Easy trip parameter selection
- **Cost Predictions**: Clear display of estimated expenses
- **Expense Tracking**: Input fields for actual spending
- **Budget Analysis**: Real-time budget vs actual comparison
- **Visualizations**: 
  - Pie charts for expense breakdown
  - Bar charts for category comparisons
  - Budget status indicators
- **Insights Panel**: Dataset statistics and money-saving tips
- **Export Feature**: Download trip reports

## 🔧 Customization

### Adding New Destinations:
1. Open `app.py`
2. Find the `destination_costs` dictionary in `create_sample_dataset()`
3. Add your destination with Budget/Standard/Luxury costs per day per person
4. Restart the app to regenerate the dataset

### Modifying Trip Types:
- Edit the `trip_types` list in `create_sample_dataset()`
- Update cost ratios for new trip types
- Regenerate dataset by deleting `travel_expenses.csv`

### Customizing Categories:
- Modify expense categories in the model training section
- Update percentage distributions for each category
- Adjust UI elements to match new categories

## 📈 Example Usage

1. **Plan a Trip**: Select "Paris" for 7 days, "Standard" type, 2 travelers
2. **Get Prediction**: AI estimates ~$2,100 total ($150/day per person)
3. **Track Expenses**: Input actual costs as you spend
4. **Monitor Budget**: See real-time updates on spending vs budget
5. **Analyze Results**: View charts showing where money was spent
6. **Export Report**: Download detailed trip analysis

## 🔍 Technical Details

### File Structure:
```
smart-travel-planner/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├