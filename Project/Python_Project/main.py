import torch
from dataset import BitcoinRegressionDataset
from network import network
import csv

x ,y = [],[]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
with open("Bitcoin Price (USD).csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    next(csv_reader)
    for row in csv_reader:
        #print(row)
        y.append(float(row[4]))
        x.append([float(row[1]),float(row[2]),float(row[3]),float(row[5]),float(row[7]),float(row[8]),float(row[9]),float(row[10])])

N = len(y)
train_x = x[:int(N*0.8)]
train_y = y[:int(N*0.8)]

val_x = x[int(N*0.8):int(N*0.9)]
val_y = y[int(N*0.8):int(N*0.9)]

test_x = x[int(N*0.9):]
test_y = y[int(N*0.9):]

trainset = BitcoinRegressionDataset(train_x,train_y)

data_loader_train = torch.utils.data.DataLoader(
    trainset,
    batch_size=16,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

valset = BitcoinRegressionDataset(val_x,val_y)

data_loader_val = torch.utils.data.DataLoader(
    valset,
    batch_size=16,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

testset = BitcoinRegressionDataset(test_x,test_y)

data_loader_test = torch.utils.data.DataLoader(
    testset,
    batch_size=16,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

network = network(8,256,256)

network = network.to(device)


optimizer = torch.optim.Adam(network.parameters(), lr=3e-4)

criterion = torch.nn.MSELoss()

for epoch in range(100):
    train_loss = 0
    val_loss = 0
    for x,y in data_loader_train:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad() #reset gráfico do gradiente
        y_pred = network(x) #previsão da rede para este x
        loss = criterion(y,y_pred) #diferença entre o que devia ter previsto e o que rede preveu 
        loss.backward() #para calcular o gradiente
        optimizer.step() #atualizar os pesos

        train_loss += loss.item()
    
    for x,y in data_loader_val: 
        x = x.to(device)
        y = y.to(device)      
        y_pred = network(x)
        loss = criterion(y,y_pred)
        

        val_loss += loss.item()
    
    train_loss = train_loss / len(train_y)
    val_loss = val_loss / len(val_y)

    print(f"Epoch {epoch+1}/100 : Train loss = {train_loss} | Val loss = {val_loss}")

test_loss = 0
for x,y in data_loader_test:
    x = x.to(device)
    y = y.to(device)       
    y_pred = network(x)
    loss = criterion(y,y_pred)

    test_loss += loss.item()

test_loss = test_loss / len(test_y)

print(f"Test loss = {test_loss}")
        
torch.save(network.state_dict(), "network.tar")


"""

passar à rede que devolver entre -1 e 1 || network.load_state_dict("network.tar") preço=network()

desnomalizar esse valor

"""
