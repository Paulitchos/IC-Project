close all; clear; clc

importfile('Bitcoin Price (USD)',' ');
whos % Diz todas as variaveis que temos no ambiente de trabalho

%Conjunto de treino
conjuntoDeTreino_x = Bp(1:188318,1:end)'; %Entradas
conjuntoDeTreino_y = Bp(1:188318,end)'; %Saídas

% ============ Validação-Start ====================
NrInstancesTrain = length(conjuntoDeTreino_y);
idx_val = randi(NrInstancesTrain,round((10/100)* NrInstancesTrain),1);


conjuntoDeTreino_x_val = conjuntoDeTreino_x(:,idx_val); %validacao
conjuntoDeTreino_y_val = conjuntoDeTreino_y(1,idx_val); %validacao

idx_all = 1:NrInstancesTrain; idx_trainTrain = setdiff(idx_all,idx_val);
conjuntoDeTreino_x_train = conjuntoDeTreino_x(:,idx_trainTrain);
conjuntoDeTreino_y_train = conjuntoDeTreino_y(idx_trainTrain);
% ============ Validação-End ========================

size(conjuntoDeTreino_x)
size(conjuntoDeTreino_y)

conjuntoDeTeste_x = Bp(188318:end,1:end)'; %Entradas
conjuntoDeTeste_y = Bp(188318:end,end)'; %Saídas

size(conjuntoDeTeste_x)
size(conjuntoDeTeste_y)

figure; hold on
subplot(5,2,1); plot(conjuntoDeTreino_x_train(1,:)); ylabel('Obrigações')
subplot(5,2,2); plot(conjuntoDeTreino_x_val(1,:));

subplot(5,2,3); plot(conjuntoDeTreino_x_train(2,:)); ylabel('Lucros por ação')
subplot(5,2,4); plot(conjuntoDeTreino_x_val(2,:));

subplot(5,2,5); plot(conjuntoDeTreino_x_train(3,:)); ylabel('Dividendos por ação')
subplot(5,2,6); plot(conjuntoDeTreino_x_val(3,:));

subplot(5,2,7); plot(conjuntoDeTreino_x_train(4,:)); ylabel('valor do índice na semana atual')
subplot(5,2,8); plot(conjuntoDeTreino_x_val(4,:));

subplot(5,2,9); plot(conjuntoDeTreino_y_train); ylabel('valor do índice na semana seguinte')
subplot(5,2,10); plot(conjuntoDeTreino_y_val);




%Treinar rede e avaliar
numero_de_neuronios = [30 30 30]; %Três camadas escondidas com 10 , 5 e 10 neurónios respetivamente
coeficiente_aprendizagem = 0.01; %learning rate - parametro importante

netMLP = train(numero_de_neuronios,coeficiente_aprendizagem,conjuntoDeTreino_x_train,conjuntoDeTreino_y_train,conjuntoDeTreino_x_val,conjuntoDeTreino_y_val);
% netRBF = Exercicio1d(conjuntoDeTreino_x_train,conjuntoDeTreino_y_train);

testY_MLP = netMLP(conjuntoDeTreino_x_val); %Prever saídas com a rede MLP
perfMLP = perform(netMLP,testY_MLP,conjuntoDeTreino_y_val); %Avaliar a performance da rede MLP


figure; hold on
plot(conjuntoDeTreino_y_val,'k','LineWidth',2)
plot(testY_MLP,'b')
plot(testY_RBF,'r')
legend('real','MLP','RBF')


