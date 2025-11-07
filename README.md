📈 SpendSight: Personal Expense Analyzer & Predictor

An interactive web app that demonstrates the 7-stage data science process. Upload your own expense file (or generate sample data) to receive an instant analysis of your spending patterns and a personalized 3-month forecast.

🚀 Try the Live App!

See the project in action right now—no installation required.

<img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Try it on Streamlit" width="200">

(Note: The app may take a moment to load if it's waking from sleep.)

✨ How It Works

This app turns your raw expense data into powerful, actionable insights in four simple steps:

Step

Action

What You Get

1.

Upload or Generate

Click "Generate Data" or upload your own CSV/Excel/JSON file.

2.

Map Your Data

Match your column names (e.g., "My Costs") to the app's fields ("Amount").

3.

Get Your Analysis

The app runs EDA and Hypothesis Tests, showing you charts on: <ul><li>Spending Over Time</li><li>Top Categories</li><li>Weekend vs. Weekday habits</li></ul>

4.

See Your Forecast

An ARIMA model trains on your data and provides an interactive forecast for the next 1-12 months.

App Demo (Placeholder)

(This is a great place to add a GIF showing the app in action!)
[Insert your GIF here, e.g., ![App Demo](demo.gif)]

🛠️ How to Run This Project Locally

Want to run the app on your own machine?

Clone the repository:

git clone [https://github.com/your-username/personal-expense-analyzer-spending-predictor.git](https://github.com/your-username/personal-expense-analyzer-spending-predictor.git)
cd personal-expense-analyzer-spending-predictor



Install the required libraries:
(You can use the provided requirements.txt file)

pip install -r requirements.txt



Run the app:
(The app uses streamlit_app_v3.py by default, but you can choose any version)

streamlit run streamlit_app_v3.py



Your browser will automatically open to http://localhost:8501.

📂 File Descriptions

streamlit_app_v3.py: (Main App) The final, most advanced Streamlit app. It includes both data generation and user upload.

Project_Notebook.py: An explanatory Python script formatted as a notebook. Great for walking through the 7 stages step-by-step in an IDE like VS Code.

personal_expense_analyzer.py: A simple, non-interactive Python script that runs all 7 stages in your terminal and opens plot windows.

requirements.txt: A list of all Python libraries needed to run the project.

📊 The 7 Stages of Data Science in This Project

This project is a practical demonstration of the complete data science lifecycle:

Business Understanding: Define the problem (users don't understand their spending) and the goal (analyze past spending and predict future spending).

Data Acquisition: Implement two methods: generating new synthetic data and loading user-provided files (CSV/Excel/JSON).

Data Preparation: Clean the data (handle nulls, convert data types) and perform feature engineering (extracting Month, Day_of_Week, Is_Weekend from the Date).

Exploratory Data Analysis (EDA): Use matplotlib and seaborn to visualize spending patterns (distributions, categorical counts, time-series plots).

Modeling: Build an ARIMA (Autoregressive Integrated Moving Average) time-series model to learn from past monthly spending data.

Model Evaluation: Test the model's accuracy against a held-back "test set" of data. Calculate key metrics like $RMSE$ (error in dollars) and $MAPE$ (average percentage error).

Deployment: Deploy the entire pipeline as an interactive Streamlit web application that anyone can use.