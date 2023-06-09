import torch
from network import network

prediction_values = torch.Tensor([57450.89000000,57475.66000000,57435.51000000,14.16831800,814059.35165094,730,7.24775100,416412.02220631])

x_max=torch.Tensor([6.48000000e+04, 6.48540000e+04 ,6.46851700e+04 ,1.86693905e+03,1.04698422e+08 ,2.91640000e+04 ,1.17949386e+03 ,5.56839455e+07])
x_min=torch.Tensor([28241.95 ,28764.23 ,28130,       0,       0,       0,       0,       0,  ])


y_max=torch.Tensor([64800.0])
y_min=torch.Tensor([28235.47])

prediction_values = ((prediction_values-x_min)/(x_max - x_min))*2 -1

network = network(8,256,256)
network.load_state_dict(torch.load("network.tar"))

preco = network(prediction_values)

preco = (preco * (y_max - y_min) + y_min)

print(f"Preço Adivinhado : {preco.item()}")