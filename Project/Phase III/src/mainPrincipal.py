from sklearn.model_selection import train_test_split
import numpy as np
import csv
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam

from dataset import max_min

x ,y = [],[]

with open("Bitcoin Price (USD).csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    next(csv_reader)
    for row in csv_reader:
        #print(row)
        y.append(float(row[4])) #saidas
        x.append([float(row[1]),float(row[2]),float(row[3]),float(row[5]),float(row[7]),float(row[8]),float(row[9]),float(row[10])]) #features que queremos (entradas)

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=0)

x_max,x_min,y_max,y_min = max_min(train_x,train_y)

model = Sequential()
model.add(Dense(256, input_shape=(8,), activation='tanh', name='C1'))
model.add(Dense(256, activation='tanh', name='C2'))
model.add(Dense(1, activation='softmax', name='output'))

model.summary()

optimizer = Adam(lr=0.001)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# Treino da rede
model.fit(train_x, train_y, verbose=2, batch_size=5, epochs=200)

# Avaliação
results = model.evaluate(test_x, test_y)
print("Resultado",results)
