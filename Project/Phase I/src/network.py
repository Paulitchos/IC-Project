import torch.nn as nn
import torch.nn.functional as F
import torch
from torchviz import make_dot

class network(nn.Module):
    #Criar uma rede com 2 camadas escondidas e uma de saída
    def __init__(self,inputSize,fc1Dim,fc2Dim):#Features, neuronios da primeira camada, neuronios da segunda camada
        super(network, self).__init__() #Devolve um objeto temporario (network)
        #inputSize é a quantidade de atributos

        self.fc1 = nn.Linear(inputSize,fc1Dim) #fc -> fully connected
        self.fc2 = nn.Linear(fc1Dim,fc2Dim)
        self.out = nn.Linear(fc1Dim,1)


    #
    def forward(self,x): #Função para percorrer cada camada
        # Passar por cada camada
        x = self.fc1(x)
        #print(self.layer[0].get_weights())
        x = self.fc2(x)
        x = self.out(x)
        #x = torch.tanh(x) # Limitar entre -1 e 1 o output
        #x = torch.softmax(x,1) 
        #x = torch.rrelu(x)
        x= torch.sigmoid(x)

        model = nn.Sequential()
        model.add_module('W0', nn.Linear(8, 16))
        model.add_module('tanh', nn.Tanh())
        model.add_module('W1', nn.Linear(16, 1))
        x = torch.randn(1, 8)
        y = model(x)
        make_dot(y.mean(), params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
        
        return x


