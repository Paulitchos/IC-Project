# Import libraries and modules
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import csv
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.callbacks import History
from keras.wrappers.scikit_learn import KerasRegressor
import pyswarms as ps


def create_model(camadas, neuronios, learnr, act_h,act_out):
    # Set up model
    model = Sequential()
    for i in range(camadas):
        model.add(Dense(neuronios, input_dim=8, activation=act_h))
    model.add(Dense(1, activation=act_out))

    # Compile model
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learnr))

    return model

# Read data from CSV file
x, y = [], []

with open("Bitcoin Price (USD).csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    next(csv_reader)
    for row in csv_reader:
        y.append(float(row[4])) # Outputs
        x.append([float(row[1]),float(row[2]),float(row[3]),float(row[5]),float(row[7]),float(row[8]),float(row[9]),float(row[10])]) # Inputs

# Convert x and y to numpy arrays
x = np.array(x)
y = np.array(y)

# Split the data into training and testing sets
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=0)

# Normalize training and testing data
scaler = MinMaxScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

# Reshape train_y to have shape (n_samples, 1)
train_y = train_y.reshape(-1, 1)

# Normalize training and testing data
train_y = scaler.fit_transform(train_y)
test_y = scaler.transform(test_y.reshape(-1, 1))


#=============== Parte Grid ===============#

param_grid = {
    'camadas': [1, 2, 3, 4],
    'neuronios': [16, 32, 64, 128],
    'learnr': [1e-3, 1e-4, 1e-5],
    'act_h': ['relu', 'tanh'],
    'act_out': ['linear', 'sigmoid']
}

model = KerasRegressor(build_fn=create_model)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(train_x, train_y)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

best_model = create_model(**best_params)
best_model.fit(train_x, train_y, epochs=50, batch_size=64)
test_loss = best_model.evaluate(test_x, test_y)