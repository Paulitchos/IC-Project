
import torch
from torch.utils.data import Dataset
import numpy as np

class BitcoinRegressionDataset(Dataset): #Recebe o Dataset do main 

    def __init__(self,x,y,x_max,x_min,y_max,y_min): 
        self.x, self.y = x,y # => guardar x e y
        
        # valores máximos e mínimos das colunas dos atributos para normalizar os valores
        self.x_max = x_max
        self.x_min= x_min
        # valor máximo e mínimo da coluna de output para normalizar os valor
        self.y_max= y_max
        self.y_min= y_min

    # Obter número de dados
    def __len__(self,):
        return len(self.y) #Devolve a quantidade de linhas do y

    # Para aceder a um indice específico como se fosse um vetor
    def __getitem__(self,index):
        # Aceder o indice para obter o valor, coverte-se em Tensor que é usado pelo pytorch para treinar a rede
        x = torch.Tensor(self.x[index]) 
        y = torch.Tensor([self.y[index]])
        # Normalizar os valores entre -1 e 1
        x = ((x-self.x_min)/(self.x_max - self.x_min))*2 -1
        y = ((y-self.y_min)/(self.y_max - self.y_min))*2 -1

        #print(y)
        return x,y

class BitcoinRegressionDataset_train(Dataset): #Recebe o Dataset do main 

    def __init__(self,x,y,x_max,x_min,y_max,y_min): 
        self.x, self.y = x,y # => guardar x e y
        self.x_max,self.x_min,self.y_max,self.y_min = x_max,x_min,y_max,y_min
    # Obter número de dados
    def __len__(self,):
        return len(self.y) #Devolve a quantidade de linhas do y

    # Para aceder a um indice específico como se fosse um vetor
    def __getitem__(self,index):
        # Aceder o indice para obter o valor, coverte-se em Tensor que é usado pelo pytorch para treinar a rede
        x = torch.Tensor(self.x[index]) 
        y = torch.Tensor([self.y[index]])
        x
        # Normalizar os valores entre -1 e 1
        x = ((x-self.x_min)/(self.x_max - self.x_min))*2 -1
        y = ((y-self.y_min)/(self.y_max - self.y_min))*2 -1

        #print(y)
        return x,y

def max_min(x,y):
    x = np.array(x) #Mete o train_x no array 
    y = np.array(y)

    print(x.shape)
    print(y.shape)

    x_max = np.max(x,axis=0)
    y_max = np.max(y)

    print(x_max)
    print(y_max)

    x_min = np.min(x,axis=0)
    y_min = np.min(y)

    print(x_min)
    print(y_min)
    
    
    
    # valores máximos e mínimos das colunas dos atributos para normalizar os valores
    x_max=torch.Tensor([x_max[0],x_max[1],x_max[2],x_max[3],x_max[4],x_max[5],x_max[6],x_max[7]])
    x_min=torch.Tensor([x_min[0],x_min[1],x_min[2],x_min[3],x_min[4],x_min[5],x_min[6],x_min[7]])

    # valor máximo e mínimo da coluna de output para normalizar os valor
    y_max=torch.Tensor([y_max])
    y_min=torch.Tensor([y_min])    

    return x_max,x_min,y_max,y_min     