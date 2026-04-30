clc;
clear

for j=1:240
    file=sprintf('%d.xlsx',j);
[num,txt,d]=xlsread(file);
n=(num(:,1))';
n(isnan(n)) = 0;
x{j,:}=n;
end
[y,txt2,d2]=xlsread('KS_Labels.xlsx');
y=categorical(y);
[m,n] = size(x) ;
P = 0.70 ;
idx = randperm(m)  ;
XTrain = x(idx(1:round(P*m)),:) ; 
XTest = x(idx(round(P*m)+1:end),:) ;
YTrain = y(idx(1:round(P*m)),:) ; 
YTest = y(idx(round(P*m)+1:end),:) ;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numObservations = numel(XTrain);
for i=1:numObservations
    sequence = XTrain{i};
    sequenceLengths(i) = size(sequence,2);
end
[sequenceLengths,idx] = sort(sequenceLengths);
XTrain = XTrain(idx);
YTrain = YTrain(idx);
miniBatchSize = 27;
inputSize = 1;
numHiddenUnits = 100;
numClasses = 5;

layers = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer]
maxEpochs = 100;
miniBatchSize = 27;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'Verbose',0, ...
    'Plots','training-progress');
net = trainNetwork(XTrain,YTrain,layers,options);
numObservationsTest = numel(XTest);
for i=1:numObservationsTest
    sequence = XTest{i};
    sequenceLengthsTest(i) = size(sequence,2);
end
[sequenceLengthsTest,idx] = sort(sequenceLengthsTest);
XTest = XTest(idx);
YTest = YTest(idx);
miniBatchSize = 27;
YPred = classify(net,XTest, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest');
acc = sum(YPred == YTest)./numel(YTest)
