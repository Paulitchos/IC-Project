# Import libraries and modules
import csv  # for reading and writing CSV files
import matplotlib.pyplot as plt  # for creating plots and charts
import numpy as np  # for numerical computing with Python
import pandas as pd  # for data manipulation and analysis
import pathlib  # for interacting with file paths in a cross-platform manner
import tensorflow as tf  # for machine learning and deep learning
from keras.layers import Dense, Activation  # for building deep learning models in TensorFlow
from keras.models import Sequential  # for building deep learning models in TensorFlow
from keras.optimizers import Adam, RMSprop  # for building deep learning models in TensorFlow
from keras.callbacks import History
from sklearn.model_selection import train_test_split, GridSearchCV  # for model selection and evaluation
from sklearn.preprocessing import MinMaxScaler  # for preprocessing data
from tensorflow import keras  # for building deep learning models in TensorFlow
from tensorflow.keras import layers  # for building deep learning models in TensorFlow
from keras.wrappers.scikit_learn import KerasRegressor

# Print TensorFlow version
print(tf.__version__)

# Define column names for the CSV file
column_names = ['Open Time','Open','High','Low','Close',
                'Close Volume', 'Time', 'Quote asset volume',
               'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume']
               

# Read in the CSV file using pandas
raw_dataset = pd.read_csv("main.csv", names=column_names,
                      na_values = "?", comment='\t', dtype='float',
                      sep=",", skipinitialspace=True)

# Copy the data from the raw dataset to a new dataframe
dataset = raw_dataset.copy()

# Print the last few rows of the dataset
dataset.tail()

dataset.pop("Open Time")
dataset.pop("Time")
# Randomly select 80% of the rows from the dataset and store them in a new dataframe
train_dataset = dataset.sample(frac=0.8,random_state=0)

# Remove the rows in the training dataset from the original dataset, leaving the remaining rows in a new dataframe
test_dataset = dataset.drop(train_dataset.index)

# Calculate statistical summary of the training dataset
train_stats = train_dataset.describe()

# Remove the "Close" column from the statistical summary
train_stats.pop("Close")

# Transpose the statistical summary so that it's in a more useful shape
train_stats = train_stats.transpose()

# Remove the "Close" column from the training dataset and store it in a new dataframe
train_labels = train_dataset.pop('Close')

# Remove the "Close" column from the testing dataset and store it in a new dataframe
test_labels = test_dataset.pop('Close')

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_dataset)
normed_train_dataset = scaler.transform(train_dataset)
normed_test_dataset = scaler.transform(test_dataset)

print(normed_train_dataset.shape)
print(train_labels.shape)

print(train_labels)

param_grid = {
    'camadas': [1, 2, 3, 4],
    'neuronios': [16, 32, 64, 128],
    'act_h': ['relu', 'tanh'],
}

def create_model(camadas, neuronios, act_h):
    # Set up model
    model = Sequential()
    model.add(Dense(neuronios, input_dim=8, activation=act_h))
    for i in range(camadas):
        model.add(Dense(neuronios,activation=act_h))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    optimizer = Adam(0.001)

    model.compile(loss='mse',optimizer=optimizer,metrics=['mae', 'mse'])

    return model

model = KerasRegressor(build_fn=create_model)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(normed_train_dataset, train_labels)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(best_params)