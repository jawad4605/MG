from flask import Flask, request, render_template, redirect, url_for, flash, session
import joblib
import pandas as pd
from scipy.stats import pearsonr, chi2_contingency
import gspread
from oauth2client.service_account import ServiceAccountCredentials

app = Flask(__name__)
app.secret_key = "secret-key"

# Load models
model_logistic = joblib.load('logistic_regression_model.pkl')
model_linear = joblib.load('linear_regression_model.pkl')

# Google Sheets API setup
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive"
]
creds = ServiceAccountCredentials.from_json_keyfile_name('D:/MG core/helpful-helper-435409-s2-f8a5d299ca06 (1).json', scope)
client = gspread.authorize(creds)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/performance')
def performance_form():
    return render_template('performance.html')

@app.route('/turnover')
def turnover_form():
    return render_template('turnover.html')

@app.route('/correlation')
def correlation_form():
    data = session.get('data')
    if data:
        df = pd.read_json(data)
        table_data = df.to_html(classes='data', header="true")
    else:
        table_data = None
    return render_template('correlation.html', table_data=table_data)

@app.route('/chi_square')
def chi_square_form():
    chi_square_data = session.get('chi_square_data')
    results = session.get('chi_square_results')
    if chi_square_data:
        df = pd.read_json(chi_square_data)
        table_data = df.to_html(classes='data', header="true")
    else:
        table_data = None
    
    return render_template('chi_square.html', table_data=table_data, results=results)

@app.route('/predict_performance', methods=['POST'])
def predict_performance():
    data = request.form
    training_hours = float(data['Training_Hours'])
    predicted_performance = model_linear.predict([[training_hours]])[0]
    return render_template('result.html', prediction_type="Performance", prediction_value=round(predicted_performance, 2))

@app.route('/predict_turnover', methods=['POST'])
def predict_turnover():
    data = request.form
    salary = float(data['Salary'])
    satisfaction = int(data['Satisfaction'])
    years_tenure = int(data['Years_Tenure'])
    features = [[salary, satisfaction, years_tenure]]
    turnover_probability = model_logistic.predict_proba(features)[0][1]
    return render_template('result.html', prediction_type="Turnover Probability", prediction_value=round(turnover_probability, 2))

@app.route('/get_data', methods=['POST'])
def get_data():
    # Get data from sheet 1 for Engagement and Productivity
    sheet1 = client.open("data").sheet1
    data1 = sheet1.get_all_records()
    df1 = pd.DataFrame(data1)
    required_columns1 = ['Engagement_Score', 'Productivity_Score']
    df1 = df1[required_columns1]

    # Store data in session for correlation
    session['data'] = df1.to_json(orient='records')
    
    flash("Correlation data fetched successfully!")
    return redirect(url_for('correlation_form'))

@app.route('/get_chi_square_data', methods=['POST'])
def get_chi_square_data():
    # Get data for Chi-Square from its respective sheet
    sheet = client.open("data").worksheet("chi_square")
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    required_columns = ['Department', 'Satisfaction']
    df = df[required_columns]

    # Store Chi-Square data in session
    session['chi_square_data'] = df.to_json(orient='records')
    
    flash("Chi-Square data fetched successfully!")
    return redirect(url_for('chi_square_form'))

@app.route('/calculate_correlation', methods=['POST'])
def calculate_correlation():
    data = session.get('data')
    if not data:
        flash("No data found. Please fetch the data first.")
        return redirect(url_for('correlation_form'))
    
    df = pd.read_json(data)
    corr, _ = pearsonr(df['Engagement_Score'], df['Productivity_Score'])
    flash(f'Pearson Correlation Coefficient: {corr:.2f}')
    return redirect(url_for('correlation_form'))

@app.route('/calculate_chi_square', methods=['POST'])
def calculate_chi_square():
    data = session.get('chi_square_data')
    if not data:
        flash("No data found. Please fetch the Chi-Square data first.")
        return redirect(url_for('chi_square_form'))
    
    df = pd.read_json(data)
    contingency_table = pd.crosstab(df['Department'], df['Satisfaction'])
    chi2, p, _, _ = chi2_contingency(contingency_table)

    # Store results in session
    session['chi_square_results'] = (chi2, p)
    
    flash(f'Chi-Square statistic: {chi2:.2f}, p-value: {p:.4f}')
    return redirect(url_for('chi_square_form'))

if __name__ == '__main__':
    app.run(debug=True)
