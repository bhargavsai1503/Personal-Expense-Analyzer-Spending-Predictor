# 📈 SpendSight: Personal Expense Analyzer & Predictor

An interactive Streamlit app that demonstrates the 7-stage data science process. Upload your own expense file (or generate sample data) to receive an instant analysis of your spending patterns and a personalized 3-month forecast.

## 🚀 Live App

[**Click here to try the live app!**](https://www.google.com/search?q=https://spendsight.streamlit.app/)

*(Note: The app may take a moment to load if it's waking from sleep.)*

## ✨ Project Highlights

* **📈 Full Data Science Pipeline:** A 7-stage project in one app, from data ingestion to predictive modeling and deployment.

* **↔️ Dual Input Modes:** Don't have a file? Use the `Generate sample data` button. Have your own?`Upload your own` (CSV, Excel, or JSON).

* **🗺️ Column Mapping:**  An intuitive interface that allows you to map your file's specific column names (e.g., "Transaction Date," "Cost") to the ones the app needs (e.g., "Date," "Amount").

* **📊 Rich EDA:** Get an instant Analyzer with interactive charts for spending over time, breakdowns by category, and distributions of transaction amounts.

* **🧪 Hypothesis Testing:** The app automatically runs live T-tests to find statistically significant insights in your data, such as whether weekend spending is really different from weekday spending.

* **🔮 Predictive Forecasting:** A powerful `ARIMA time-series model` trains on your data to generate a "Predictor" forecast for your total spending over the next 1-12 months.

* **💬 Simple Explanations:** All results (including p-values and model accuracy) are translated into plain English so you know exactly what they mean.

## 🏃 How to Run Locally

1. **Clone the repository:**



        git clone https://github.com/bhargav-sai-tanguturi/personal-expense-analyzer-spending-predictor.git 
        cd personal-expense-analyzer-spending-predictor


2. **Install the required libraries:**



        pip install -r requirements.txt


3. **Run the app:**



        streamlit run streamlit_app.py


Your browser will automatically open to `http://localhost:8501`.

## 📂 File Descriptions

* `streamlit_app.py`: **(Main App)** The final, most advanced Streamlit app.

* `Explanatory_Notebook.ipynb`: An explanatory Python script (like a Jupyter notebook) for walking through the 7 stages.



* `requirements.txt`: A list of all Python libraries needed to run the project.

