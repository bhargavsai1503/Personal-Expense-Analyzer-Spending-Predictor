# --- Streamlit App: Personal Expense Analyzer (V3 Combined) ---
#
# To run this app:
# 1. Save this file as 'streamlit_app_v3.py'
# 2. Open your terminal in the same directory.
# 3. Run: pip install streamlit pandas numpy scipy scikit-learn statsmodels matplotlib seaborn openpyxl
# 4. Run: streamlit run streamlit_app_v3.py

import streamlit as st
import pandas as pd
import numpy as np
import warnings
from io import StringIO, BytesIO
import random
from datetime import datetime, timedelta

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Hypothesis Testing
from scipy import stats

# Modeling
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Suppress common warnings
warnings.filterwarnings("ignore")

# --- App Title ---
st.set_page_config(layout="wide")
st.title("📈SpendSight: Personal Expense Analyzer & Predictor ")
st.write("""
This app demonstrates the 7-stage Data Science process. You can either **generate
synthetic data** or **upload your own expense file** (CSV, Excel, or JSON).
The app will then analyze spending patterns and forecast future monthly spending.
""")

# ======================================================================
# STAGE 1 & 2: DATA INPUT (GENERATE OR UPLOAD)
# ======================================================================

# --- Helper Functions for Data Ingestion ---

@st.cache_data
def generate_data():
    """Generates a unique, synthetic dataset for the project."""
    NUM_ROWS = 1500
    START_DATE = datetime(2023, 1, 1)
    END_DATE = datetime(2024, 12, 31)

    categories = {
        'Food': ['Groceries', 'Dining Out', 'Coffee', 'Delivery'],
        'Transport': ['Gas', 'Public Transit', 'Taxi/Rideshare'],
        'Housing': ['Rent', 'Utilities', 'Insurance'],
        'Personal': ['Shopping', 'Gym', 'Entertainment', 'Subscription'],
        'Other': ['Gifts', 'Health', 'Miscellaneous']
    }
    payment_methods = ['Credit Card', 'Debit Card', 'Cash', 'Online Transfer']

    data = []
    date_range = (END_DATE - START_DATE).days

    for i in range(NUM_ROWS):
        trans_date = START_DATE + timedelta(days=random.randint(0, date_range))
        main_cat = random.choice(list(categories.keys()))
        amount = 0
        if main_cat == 'Housing': amount = random.uniform(500, 2000)
        elif main_cat == 'Food': amount = random.uniform(5, 150)
        elif main_cat == 'Transport': amount = random.uniform(10, 70)
        elif main_cat == 'Personal': amount = random.uniform(15, 250)
        else: amount = random.uniform(10, 100)
        
        if main_cat == 'Housing' and random.random() > 0.5: # Simulate Rent
            amount = 1500 + random.uniform(-50, 50)
            trans_date = trans_date.replace(day=1)
            
        payment = random.choice(payment_methods)
        
        data.append([trans_date, round(amount, 2), main_cat, payment])

    df = pd.DataFrame(data, columns=['Date', 'Amount', 'Category', 'Payment_Method'])
    df = df.drop_duplicates(subset=['Date', 'Category'], keep='first')
    df = df.sort_values(by='Date').reset_index(drop=True)
    return df

def process_generated_data(df):
    """Applies feature engineering to the generated data."""
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day_of_Week'] = df['Date'].dt.day_name()
    df['Is_Weekend'] = df['Day_of_Week'].isin(['Saturday', 'Sunday'])
    return df

def read_uploaded_file(uploaded_file):
    """Reads various file types into a DataFrame."""
    try:
        if uploaded_file.type == "text/csv":
            return pd.read_csv(uploaded_file)
        elif uploaded_file.type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
            return pd.read_excel(uploaded_file)
        elif uploaded_file.type == "application/json":
            return pd.read_json(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

# --- Session State Initialization ---
if 'raw_df' not in st.session_state:
    st.session_state.raw_df = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

# --- UI for Data Input ---
st.header("Stage 1 & 2: Business Need & Data Input")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Option 1: Generate Data")
    st.markdown("Click here to generate a synthetic 2-year expense dataset for analysis.")
    if st.button("🚀 Generate & Analyze Data"):
        with st.spinner("Generating synthetic data..."):
            df_gen = generate_data()
            st.session_state.processed_df = process_generated_data(df_gen)
            st.session_state.processing_complete = True
            st.session_state.raw_df = None # Clear raw df if it exists
            st.success("Generated data is ready! Scroll down to see analysis.")

with col2:
    st.subheader("Option 2: Upload Your File")
    st.markdown("Upload your own expense file (CSV, XLSX, JSON) for a personal analysis.")
    uploaded_file = st.file_uploader(
        "Choose your expense file",
        type=["csv", "xlsx", "xls", "json"]
    )
    if uploaded_file is not None:
        st.session_state.raw_df = read_uploaded_file(uploaded_file)
        st.session_state.processing_complete = False
        st.session_state.processed_df = None

# --- Column Mapper (for uploaded files) ---
if st.session_state.raw_df is not None:
    st.success("File Uploaded! Please map your columns.")
    st.dataframe(st.session_state.raw_df.head())
    
    st.subheader("Column Mapping")
    st.write("Please map your file's columns to the fields required for analysis.")
    
    df_raw = st.session_state.raw_df
    columns = df_raw.columns.tolist()
    
    map_col1, map_col2 = st.columns(2)
    with map_col1:
        date_col = st.selectbox("Select your 'Date' column:", columns)
        amount_col = st.selectbox("Select your 'Amount' column:", columns, index=min(1, len(columns)-1))
    with map_col2:
        category_col = st.selectbox("Select your 'Category' column:", columns, index=min(2, len(columns)-1))
        payment_col = st.selectbox("Select your 'Payment Method' column:", columns, index=min(3, len(columns)-1))

    if st.button("📊 Analyze Uploaded Data"):
        with st.spinner("Processing your data..."):
            # STAGE 3 (Processing) for Uploaded Data
            try:
                processed_df = df_raw[[date_col, amount_col, category_col, payment_col]].copy()
                processed_df.columns = ['Date', 'Amount', 'Category', 'Payment_Method']
                
                processed_df['Date'] = pd.to_datetime(processed_df['Date'])
                processed_df['Amount'] = pd.to_numeric(processed_df['Amount'])
                processed_df = processed_df.dropna()

                # Feature Engineering
                processed_df['Year'] = processed_df['Date'].dt.year
                processed_df['Month'] = processed_df['Date'].dt.month
                processed_df['Day_of_Week'] = processed_df['Date'].dt.day_name()
                processed_df['Is_Weekend'] = processed_df['Day_of_Week'].isin(['Saturday', 'Sunday'])
                
                st.session_state.processed_df = processed_df
                st.session_state.processing_complete = True
                st.success("Uploaded data is ready! Scroll down to see analysis.")
            
            except Exception as e:
                st.error(f"Error processing data: {e}")
                st.session_state.processing_complete = False


# ======================================================================
# STAGES 3-7: ANALYSIS PIPELINE
# This section runs only if data is successfully processed
# ======================================================================

if st.session_state.processing_complete and st.session_state.processed_df is not None:
    
    df_final = st.session_state.processed_df
    
    st.header("Stage 3: Data Analysis")
    st.write("Data after cleaning and feature engineering:")
    st.dataframe(df_final.head())

    # ======================================================================
    # STAGE 4: EXPLORATORY DATA ANALYSIS (EDA) & HYPOTHESIS TESTING
    # ======================================================================

    st.header("Stage 4: Exploratory Data Analysis & Hypothesis Testing")

    st.subheader("4.1 Key Visualizations")
    col1, col2 = st.columns(2)

    with col1:
        st.write("#### Distribution of Transaction Amounts")
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        # Plotting a subset if data is very large to avoid warning
        sample_size = min(5000, len(df_final))
        sns.histplot(df_final['Amount'].sample(sample_size), bins=50, kde=True, ax=ax1)
        ax1.set_title('Distribution of Transaction Amounts')
        ax1.set_xlabel('Amount ($)')
        st.pyplot(fig1)

    with col2:
        st.write("#### Number of Transactions by Category")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        sns.countplot(y='Category', data=df_final, order=df_final['Category'].value_counts().index[:15], ax=ax2)
        ax2.set_title('Transactions by Category (Top 15)')
        ax2.set_xlabel('Number of Transactions')
        st.pyplot(fig2)

    # --- NEW PLOT ADDED HERE ---
    st.write("#### Distribution of Amount by Category (Box Plot)")
    fig4, ax4 = plt.subplots(figsize=(12, 8))
    # fliersize=0 hides outliers that are beyond the xlim
    sns.boxplot(y='Category', x='Amount', data=df_final, ax=ax4, fliersize=0) 
    ax4.set_title('Distribution of Amount by Category (Zoomed In)')
    ax4.set_xlabel('Amount ($)')
    ax4.set_ylabel('Category')
    ax4.set_xlim(0, 1000) # Zoom in as requested
    st.pyplot(fig4)
    st.caption("Note: X-axis is zoomed in to $0 - $1,000 to show detail on common transactions, excluding high outliers like rent.")
    # --- END OF NEW PLOT ---

    st.write("#### Total Monthly Spending Over Time")
    monthly_spending = df_final.resample('M', on='Date')['Amount'].sum()
    fig3, ax3 = plt.subplots(figsize=(14, 5))
    monthly_spending.plot(ax=ax3)
    ax3.set_title('Total Monthly Spending Over Time')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Total Spending ($)')
    st.pyplot(fig3)

    st.subheader("4.2 Hypothesis Testing")
    col_h1, col_h2 = st.columns(2)

    with col_h1:
        st.write("#### H1: Weekend vs. Weekday Spending (T-test)")
        st.write("**H0:** The average spending amount is the same.")
        st.write("**Ha:** The average spending amount is different.")
        
        weekend_spending = df_final[df_final['Is_Weekend'] == True]['Amount']
        weekday_spending = df_final[df_final['Is_Weekend'] == False]['Amount']
        
        if len(weekend_spending) > 1 and len(weekday_spending) > 1:
            t_stat, p_value_ttest = stats.ttest_ind(weekend_spending, weekday_spending, equal_var=False)
            st.write(f"**P-value: {p_value_ttest:.4f}**")
            if p_value_ttest < 0.05:
                st.error("Result: REJECT H0. There IS a significant difference in spending.")
                st.write("*(In words: The p-value is very small, which means it's highly unlikely this spending difference is due to random chance. Weekends and weekdays have different spending patterns.)*")
            else:
                st.success("Result: FAIL TO REJECT H0. There is NO significant difference.")
                st.write("*(In words: The p-value is high, which means we don't have enough statistical evidence to say that spending on weekends is any different from weekdays.)*")
        else:
            st.warning("Not enough data for both weekdays and weekends to run T-test.")

    with col_h2:
        st.write("#### H2: 'Food' vs 'Personal' Spending (T-test)")
        st.write("**H0:** The average transaction amount is the same.")
        st.write("**Ha:** The average transaction amount is different.")
        
        # Check if categories exist
        if 'Food' in df_final['Category'].values and 'Personal' in df_final['Category'].values:
            food_spending = df_final[df_final['Category'] == 'Food']['Amount']
            personal_spending = df_final[df_final['Category'] == 'Personal']['Amount']
            
            if len(food_spending) > 1 and len(personal_spending) > 1:
                t_stat_cat, p_value_ttest_cat = stats.ttest_ind(food_spending, personal_spending, equal_var=False)
                st.write(f"**P-value: {p_value_ttest_cat:.4f}**")
                if p_value_ttest_cat < 0.05:
                    st.error("Result: REJECT H0. There IS a significant difference in spending.")
                    st.write("*(In words: The average transaction amount for 'Food' and 'Personal' is statistically different.)*")
                else:
                    st.success("Result: FAIL TO REJECT H0. There is NO significant difference.")
                    st.write("*(In words: We cannot conclude there is a difference in average transaction amount between 'Food' and 'Personal'.)*")
            else:
                st.warning("Not enough data for one or both categories to run T-test.")
        else:
            st.warning("Data does not contain both 'Food' and 'Personal' categories for comparison.")


    # ======================================================================
    # STAGE 5 & 6: MODELING & EVALUATION
    # ======================================================================

    st.header("Stage 5 & 6: Modeling & Evaluation (Spending Predictor)")
    st.write("""
    We will build an **ARIMA Time-Series Model** to forecast total monthly spending.
    """)

    # We use the 'monthly_spending' data from EDA
    if len(monthly_spending) < 12:
        st.error(f"Not enough monthly data ({len(monthly_spending)} months) to build a forecast model. At least 12 months are recommended.")
    else:
        train_size = int(len(monthly_spending) * 0.8)
        train_data, test_data = monthly_spending[:train_size], monthly_spending[train_size:]

        st.write(f"Training on {len(train_data)} months, testing on {len(test_data)} months.")

        try:
            with st.spinner("Training ARIMA model... This may take a moment."):
                model_arima = ARIMA(train_data, order=(1, 1, 1))
                model_arima_fit = model_arima.fit()
                
                predictions_arima = model_arima_fit.forecast(steps=len(test_data))
                
                future_steps = st.slider("Select number of months to forecast:", 1, 12, 3)
                future_forecast = model_arima_fit.forecast(steps=len(test_data) + future_steps)
                
            st.success("Model training complete!")

            # --- Evaluation ---
            rmse_arima = np.sqrt(mean_squared_error(test_data, predictions_arima))
            mean_spending = test_data.mean()
            mape_arima = np.mean(np.abs((test_data - predictions_arima) / test_data)) * 100
            
            st.subheader("Model Evaluation (on Test Data)")
            st.metric(label="Test Data Mean Spending", value=f"${mean_spending:,.2f}")
            st.metric(label="ARIMA RMSE (Error in dollars)", value=f"${rmse_arima:,.2f}")
            st.metric(label="ARIMA MAPE (Avg. % Error)", value=f"{mape_arima:.2f}%")
            
            st.subheader("Model Interpretation (in words)")
            st.write(f"""
            - **Accuracy:** Our model forecasts your total monthly spending with an average accuracy of **{(100 - mape_arima):.2f}%**.
            - **Error (Percentage):** This means the predictions are, on average, only off by **{mape_arima:.2f}%**.
            - **Error (Dollars):** The typical error in the prediction is about **${rmse_arima:,.2f}** per month (this is the $RMSE$).
            """)

            # --- Plot Forecast ---
            st.subheader("Monthly Spending Forecast")
            fig_forecast, ax_forecast = plt.subplots(figsize=(14, 7))
            ax_forecast.plot(train_data, label='Train')
            ax_forecast.plot(test_data, label='Actual Test Data', color='orange')
            ax_forecast.plot(future_forecast, label='Forecast', color='green', linestyle='--')
            ax_forecast.set_title('ARIMA Forecast vs. Actuals')
            ax_forecast.legend()
            st.pyplot(fig_forecast)
            
            st.write("#### Forecasted Values:")
            st.dataframe(future_forecast.to_frame(name='Forecasted Spending').tail(future_steps))

        except Exception as e:
            st.error(f"An error occurred during model training: {e}")
            st.warning("Model training can fail if data has no variance or is too short. Try uploading a different file.")

    # ======================================================================
    # STAGE 7: COMMUNICATION & DEPLOYMENT
    # ======================================================================

    st.header("Stage 7: Communication & Deployment")
    st.success("""
    **Deployment Complete!** This Streamlit application itself is the deployment.

    It provides an interactive way for a user (or stakeholder) to:
    1.  **Choose** their data source (generate or upload).
    2.  **Map** their data columns for analysis.
    3.  See key analytical insights from the EDA.
    4.  Understand the results of the hypothesis tests.
    5.  Interact with the predictive model and see a personalized forecast.
    """)