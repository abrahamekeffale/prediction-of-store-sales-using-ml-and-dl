# scripts/model_serving.py
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib

app = Flask(__name__)

# Load the trained LSTM model
MODEL_PATH = '../models/lstm_model.h5'  # Adjust the path if needed
model = tf.keras.models.load_model(MODEL_PATH)

# Load scaler to preprocess data (e.g., MinMaxScaler)
SCALER_PATH = '../models/scaler.pkl'  # Assuming a scaler was saved separately
scaler = joblib.load(SCALER_PATH)

@app.route('/')
def home():
    return "Welcome to the Sales Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the request
        input_data = request.get_json()
        
        # Extract and convert data into numpy array
        data = np.array(input_data['data']).reshape(1, -1)  # reshape to be 2D
        
        # Scale the data (assuming data is in the same shape as training)
        scaled_data = scaler.transform(data)

        # Make predictions using the model
        prediction = model.predict(scaled_data)
        
        # Inverse scale the prediction
        prediction_rescaled = scaler.inverse_transform(prediction)
        
        # Return the prediction as a response
        return jsonify({'prediction': prediction_rescaled.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
