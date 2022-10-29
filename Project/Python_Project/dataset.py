
import torch
from torch.utils.data import Dataset
import numpy as np

class BitcoinRegressionDataset(Dataset):

    def __init__(self,x,y):
        self.x, self.y = x,y # => guardar x e y

        
        
        """
        x = np.array(self.x)
        y = np.array(self.y)

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
        """

        # valores máximos e mínimos das colunas dos atributos para normalizar os valores
        self.x_max=torch.Tensor([6.48000000e+04, 6.48540000e+04 ,6.46851700e+04 ,1.86693905e+03,1.04698422e+08 ,2.91640000e+04 ,1.17949386e+03 ,5.56839455e+07])
        self.x_min=torch.Tensor([28241.95 ,28764.23 ,28130,       0,       0,       0,       0,       0,  ])

        # valor máximo e mínimo da coluna de output para normalizar os valor
        self.y_max=torch.Tensor([64800.0])
        self.y_min=torch.Tensor([28235.47])

    # Obter número de dados
    def __len__(self,):
        return len(self.y)

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

    
                