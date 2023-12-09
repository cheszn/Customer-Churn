from flask import Flask, render_template, request
import numpy as np
import xgboost as xgb

app = Flask(__name__)

# Load your trained XGBoost model
xgboost_model = xgb.XGBClassifier()  # replace with the actual model loading code

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract values from HTML form
    features = []
    for i in range(1, 32):  # Assuming 31 features
        feature_value = float(request.form.get(f'feature{i}'))
        features.append(feature_value)

    # Convert the list of features to a numpy array
    features_array = np.array(features).reshape(1, -1)  # Reshape to a 2D array

    # Make prediction
    prediction = xgboost_model.predict(features_array)[0]

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
