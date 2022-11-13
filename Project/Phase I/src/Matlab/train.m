function netMLP = train(numero_de_neuronios,coeficiente_aprendizagem,conjuntoDeTreino_x,conjuntoDeTreino_y,conjuntoDeTeste_x,conjuntoDeTeste_y)


trainFcn = 'trainlm';
% Inicializar a rede
netMLP = feedforwardnet(numero_de_neuronios,trainFcn); %Está a colocar números aleatórios nos pesos (W's)

% Treinar a rede
netMLP = train(netMLP,conjuntoDeTreino_x,conjuntoDeTreino_y);


view(netMLP)





