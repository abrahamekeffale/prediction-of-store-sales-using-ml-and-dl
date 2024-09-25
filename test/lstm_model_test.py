import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the preprocessed dataset (same scaling applied during training)
data = pd.read_csv('data/data.csv', parse_dates=['Date'], index_col='Date')

# Ensure the data is sorted by Date
data.sort_index(inplace=True)

# Select the 'Sales' column which was used as the target
sales_data = data['Sales'].values.reshape(-1, 1)

# Load the trained model
model = load_model('models/lstm_model.h5')  # Assuming you saved the model as lstm_model.h5 in the 'models' folder

# Preprocess data (scaling) to match the training process
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_sales_data = scaler.fit_transform(sales_data)

# Define the window size (same as the one used in training)
window_size = 60

# Prepare the test data by taking the last 60 days
test_sequence = scaled_sales_data[-window_size:]  # Last 60 days

# Reshape the test sequence to (1, window_size, 1) to fit the LSTM model input format
test_sequence = np.reshape(test_sequence, (1, window_size, 1))

# Predict the next sales
predicted_sales_scaled = model.predict(test_sequence)

# Inverse scale the prediction back to the original sales scale
predicted_sales = scaler.inverse_transform(predicted_sales_scaled)

print(f"Predicted Sales for the next day: {predicted_sales[0][0]}")

# Plot the actual vs predicted sales for visualization 
plt.figure(figsize=(10, 6))
plt.plot(sales_data[-window_size:], label='Actual Sales (Last 60 Days)', color='blue')
plt.axhline(y=predicted_sales[0][0], color='red', linestyle='--', label='Predicted Sales (Next Day)')
plt.title('Actual Sales vs Predicted Sales')
plt.xlabel('Days')
plt.ylabel('Sales')
plt.legend()
plt.show()
