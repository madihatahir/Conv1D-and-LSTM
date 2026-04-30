clc;
clear;

% [num1,txt1,d1]= xlsread('TextData.xlsx');
%     text=txt1(2:end,:);
%     text = erasePunctuation(text);
% text = lower(text);
% text = tokenizedDocument(text);
% embeddingDimension = 100;
% embeddingEpochs = 50;
% emb = trainWordEmbedding(text, ...
%     'Dimension',embeddingDimension, ...
%     'NumEpochs',embeddingEpochs, ...
%     'Verbose',0);
% sequenceLength = 75;
% documentsTruncated = docfun(@(words) words(1:min(sequenceLength,end)),text);
% X      = doc2sequence(emb,documentsTruncated);
% X=X';
% for i = 1:numel(X)
%     XText{i} = leftPad(X{i},sequenceLength);
% end
% 
% %%%%%%%%%%%%%%%%%%% Key Strokes %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% for j=1:240
%     file=sprintf('%d.xlsx',j);
% [num,txt,d]=xlsread(file);
% 
% XKS{j,:}=d(:,1:2);
% end
% XKS=XKS';
% 
% Data = [XText;XKS]';
% save('MergedData.mat','Data');
load MergedData.mat
txt=Data;
[y,txt2,d2]= xlsread('KS_Labels.xlsx');
y=categorical(y);
 cvp = cvpartition(y,'Holdout',0.1);
XTrain = txt(training(cvp),:);
XTest = txt(test(cvp),:);
YTrain  =y(training(cvp),:);
YTest=y(test(cvp),:);
layers = [ ...
    sequenceInputLayer(240)
    bilstmLayer(512,'OutputMode','last')
    fullyConnectedLayer(5)
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
