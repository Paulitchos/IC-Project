import torch
#import keras
from network import network
#from keras.models import Sequential
#from keras.layers import Dense
#from ann_visualizer.visualize import ann_viz

from main import x_max,x_min,y_max,y_minx_max,x_min,y_max,y_min

prediction_values = torch.Tensor([56958.59000000,56996.57000000,56956.87000000,32.88001500,1873152.78176158,1083,21.40647800,1219485.98343767])

prediction_values = ((prediction_values-x_min)/(x_max - x_min))*2 -1

network = network(8,256,256)
network.load_state_dict(torch.load("network_1.tar"))

preco = network(prediction_values)

preco = (((preco + 1)/2) * (y_max - y_min) + y_min)

print(f"Preço Adivinhado : {preco.item()}")

#ann_viz(network, view=True, filename="network.gv", title="Neural Network")
