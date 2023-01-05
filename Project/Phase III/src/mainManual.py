# Import libraries and modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import csv
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.callbacks import History
from keras.wrappers.scikit_learn import KerasRegressor
import pandas as pd

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

#=============== Parte Manual ===============#
neuronios = 32
act_h = "relu"
act_out = "sigmoid"
learnr = 1
loss_method ="logcosh"
camadas = 1
epochs_n = 30

# Set up neural network model
model = Sequential()
model.add(Dense(neuronios, input_shape=(8,), activation= act_h, name='C1'))
#model.add(Dense(neuronios, activation=act_h, name='C2'))
#model.add(Dense(neuronios, activation=act_h, name='C3'))
#model.add(Dense(neuronios, activation=act_h, name='C4'))
#model.add(Dense(neuronios, activation=act_h, name='C5'))
#model.add(Dense(neuronios, activation=act_h, name='C6'))
model.add(Dense(1, activation=act_out, name='output'))

model.summary()

# Compile model
optimizer = Adam(learning_rate= learnr)
model.compile(optimizer, loss=loss_method, metrics=['mse'])
print(model.summary())

# Set up callbacks
history = History()

# Train model
model.fit(train_x, train_y, verbose=2, batch_size=5, epochs= epochs_n, callbacks=[history])

# Evaluate model
results = model.evaluate(test_x, test_y)
print("Resultado",results)

# Print average loss over all epochs
print("Average loss:", np.mean(history.history['mse']))

filename = 'modelmanual_c'+ str(camadas) +'_n'+ str(neuronios) +'_actn'+str(act_h) +'_acto'+str(act_out) +'_lr' + str(learnr) + '_loss' + str(loss_method) +'_ep' + str(epochs_n)
model.save(filename, save_format='tf')

