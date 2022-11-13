import torch.nn as nn
import torch.nn.functional as F
import torch

class network(nn.Module):
    #Criar uma rede com 2 camadas escondidas e uma de saída
    def __init__(self,inputSize,fc1Dim,fc2Dim):
        super(network, self).__init__()
        #inputSize é a quantidade de atributos
        self.fc1 = nn.Linear(inputSize,fc1Dim) #fc -> fully connected
        self.fc2 = nn.Linear(fc1Dim,fc2Dim)
        self.out = nn.Linear(fc2Dim,1)

    #
    def forward(self,x):
        # Passar por cada camada
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        x = torch.tanh(x) # Limitir entre -1 e 1 o output

        return x

    