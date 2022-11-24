import torch
from sklearn.model_selection import train_test_split
from network import network
from dataset import max_min
import csv

x ,y = [],[]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
with open("Bitcoin Price (USD).csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    next(csv_reader)
    for row in csv_reader:
        #print(row)
        y.append(float(row[4])) #saidas
        x.append([float(row[1]),float(row[2]),float(row[3]),float(row[5]),float(row[7]),float(row[8]),float(row[9]),float(row[10])]) #features que queremos (entradas)

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=0)

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.1, random_state=0)

x_max,x_min,y_max,y_min = max_min(train_x,train_y)
prediction_values = torch.Tensor([56958.59000000,56996.57000000,56956.87000000,32.88001500,1873152.78176158,1083,21.40647800,1219485.98343767])

prediction_values = ((prediction_values-x_min)/(x_max - x_min))*2 -1

network = network(8,256,256)
network.load_state_dict(torch.load("network_MAE_2Lay_256_Tanh_0_00173_.tar"))

preco = network(prediction_values)

preco = (((preco + 1)/2) * (y_max - y_min) + y_min)

print(f"Preço Adivinhado : {preco.item()}")

#ann_viz(network, view=True, filename="network.gv", title="Neural Network")
