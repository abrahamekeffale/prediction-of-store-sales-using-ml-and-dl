import pandas as pd

def load_train_data():
    train_data = pd.read_csv(r'C:\Users\HP\week 4\data\train.csv')
    return train_data

def load_test_data():
    test_data = pd.read_csv(r'C:\Users\HP\week 4\data\test.csv')
    return test_data
