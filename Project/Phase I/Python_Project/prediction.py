import torch
import keras
from network import network
from keras.models import Sequential
from keras.layers import Dense
from ann_visualizer.visualize import ann_viz

prediction_values = torch.Tensor([56958.59000000,56996.57000000,56956.87000000,32.88001500,1873152.78176158,1083,21.40647800,1219485.98343767])
x_max=torch.Tensor([6.48000000e+04, 6.48540000e+04 ,6.46851700e+04 ,1.86693905e+03,1.04698422e+08 ,2.91640000e+04 ,1.17949386e+03 ,5.56839455e+07])
x_min=torch.Tensor([28241.95 ,28764.23 ,28130,       0,       0,       0,       0,       0,  ])


y_max=torch.Tensor([64800.0])
y_min=torch.Tensor([28235.47])

prediction_values = ((prediction_values-x_min)/(x_max - x_min))*2 -1

network = network(8,256,256)
network.load_state_dict(torch.load("network.tar"))

preco = network(prediction_values)

preco = (((preco + 1)/2) * (y_max - y_min) + y_min)

print(f"Preço Adivinhado : {preco.item()}")

#ann_viz(network, view=True, filename="network.gv", title="Neural Network")
