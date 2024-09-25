# prediction-of-store-sales-using-ml-and-dl
my final week 4 project
# Rossmann Store Sales Prediction with LSTM

## Overview
This project aims to predict future sales of Rossmann stores using a Long Short Term Memory (LSTM) model, which is a type of Recurrent Neural Network (RNN) designed for time series prediction. The model is built using TensorFlow/Keras and focuses on using historical sales data to predict future sales.

## Project Structure
data/
data.csv : The Rossmann Store Sales dataset used for model training.
notebooks/
LSTM_model.ipynb : The Jupyter notebook containing the LSTM model and training code.
EDA.ipynb : Exploratory Data Analysis (EDA) of the dataset.
scripts/
data_preprocessing.py : Script for loading and preprocessing the data (scaling, creating sequences).
model_training.py : Script for building and training the LSTM model.
README.md : Project overview and instructions.

## Dataset
The dataset contains sales data of Rossmann stores and includes features such as:
- **Store** : Store ID
- **Date** : The date of the sales record
- **Sales** : The target variable we aim to predict
- **Customers** : Number of customers for that day
- **Promo** : Indicates if a store is running a promotion
- **StateHoliday** : Indicates whether the day is a state holiday
- **SchoolHoliday** : Indicates if the day is a school holiday

You can download the dataset [here](https://www.kaggle.com/c/rossmann-store-sales/data) or use your local version.

## Requirements
- Python 3.x
- TensorFlow/Keras
- Pandas
- Numpy
- Matplotlib
- Scikit-learn

Install the necessary libraries using pip:

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib
Model Architecture
We use a two-layer LSTM model for sales prediction. The model architecture includes:

Input Layer: Takes the past 60 days of sales data as input.
LSTM Layer 1: First LSTM layer with 50 units, return_sequences=True to pass the output to the next LSTM layer.
LSTM Layer 2: Second LSTM layer with 50 units.
Dense Output Layer: Predicts the next sales value based on the LSTM output.
Model Training
The model is trained using the Mean Squared Error (MSE) loss function and the Adam optimizer. We use a sliding window approach to generate sequences of 60 past days to predict the next day's sales.

Key Steps:
Data Preprocessing:
Load and preprocess the sales data (scaling and windowing).
Model Training:
Train the LSTM model on the prepared data.
Evaluation:
Evaluate the model's performance using the validation data and visualize the training and validation loss.
Usage
1. Data Preprocessing
Ensure that your dataset is placed in the data/ directory. Run the following command to preprocess the data and create sequences:
python scripts/data_preprocessing.py
2. Model Training
To train the LSTM model, execute the following:
python scripts/model_training.py
The model will automatically train on the preprocessed dataset, and the training/validation loss will be plotted after each epoch.

3. Predicting Future Sales
After training, you can use the model to predict future sales by running predictions on the last sequence in the dataset or custom sequences.

Results
After training the LSTM model for a few epochs, the model should be able to predict sales with reasonable accuracy. You can tune the number of epochs, batch size, and model architecture to improve performance.

Performance Tuning
Window Size: You can adjust the window size (number of past days used for prediction) to optimize the model's performance.
LSTM Layers: Experiment with deeper LSTM layers and units if you have more computational resources.
Batch Size and Epochs: These hyperparameters can significantly impact the model's training performance.
Acknowledgments
Rossmann Store Sales Dataset from Kaggle.
TensorFlow/Keras for providing an excellent framework for building deep learning models.
License
This project is licensed under the MIT License - see the LICENSE file for details.

---

### Key Sections:
- **Overview**: A brief introduction to the project.
- **Project Structure**: Explanation of the directory and file structure.
- **Dataset**: Information about the dataset and its features.
- **Requirements**: Dependencies needed for running the project.
- **Model Architecture**: Description of the LSTM model structure.
- **Model Training**: How to run the training process.
- **Results**: What to expect from the model.
- **Performance Tuning**: Tips for improving the model.
- **License**: Specify licensing terms for your project.

Feel free to modify the content to fit any additional information or project-specific details!


