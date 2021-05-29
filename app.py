from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import sklearn
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = joblib.load("student_grades_prediction_model.pkl")

@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    
    input_features = [float(x) for x in request.form.values()]
    value = np.array(input_features)
    
    if input_features[0] <0 or input_features[0] >24:
        return render_template('index.html', prediction_text='Please enter valid hours between 1 and 24')
    
    output = model.predict([value])[0][0].round(2)
    return render_template('index.html', prediction_text = f"You will likely receive a grade of {output}% if you study {input_features} hours per day!")

if __name__ == '__main__':
    app.debug = True
    app.run()