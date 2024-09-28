from flask import Flask, request, jsonify, render_template, redirect, url_for
import joblib

app = Flask(__name__)

# Load the models
model_logistic = joblib.load('logistic_regression_model.pkl')
model_linear = joblib.load('linear_regression_model.pkl')

@app.route('/')
def home():
    return render_template('home.html')  # Show the main page with two buttons

@app.route('/performance')
def performance_form():
    return render_template('performance.html')  # Show the performance prediction form

@app.route('/turnover')
def turnover_form():
    return render_template('turnover.html')  # Show the turnover prediction form

@app.route('/predict_performance', methods=['POST'])
def predict_performance():
    data = request.form
    training_hours = float(data['Training_Hours'])
    
    # Predict using the linear regression model
    predicted_performance = model_linear.predict([[training_hours]])[0]
    
    return render_template('result.html', prediction_type="Performance", prediction_value=round(predicted_performance, 2))

@app.route('/predict_turnover', methods=['POST'])
def predict_turnover():
    data = request.form
    salary = float(data['Salary'])
    satisfaction = int(data['Satisfaction'])
    years_tenure = int(data['Years_Tenure'])
    
    # Predict using the logistic regression model
    features = [[salary, satisfaction, years_tenure]]
    turnover_probability = model_logistic.predict_proba(features)[0][1]  # Probability of turnover
    
    return render_template('result.html', prediction_type="Turnover Probability", prediction_value=round(turnover_probability, 2))

if __name__ == '__main__': 
    app.run(debug=True)
