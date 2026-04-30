clc;
clear;
[num1,txt1,d1]= xlsread('TextData.xlsx');
    txt=txt1(2:end,:);
[y,txt2,d2]= xlsread('KS_Labels.xlsx');
y=categorical(y);
   cvp = cvpartition(y,'Holdout',0.1);
textDataTrain = txt(training(cvp),:);
textDataTest = txt(test(cvp),:);
YTrain  =y(training(cvp),:);
YTest=y(test(cvp),:);
textDataTrain = erasePunctuation(textDataTrain);
textDataTrain = lower(textDataTrain);
documentsTrain = tokenizedDocument(textDataTrain);
embeddingDimension = 100;
embeddingEpochs = 50;

emb = trainWordEmbedding(documentsTrain, ...
    'Dimension',embeddingDimension, ...
    'NumEpochs',embeddingEpochs, ...
    'Verbose',0)
sequenceLength = 75;
documentsTruncatedTrain = docfun(@(words) words(1:min(sequenceLength,end)),documentsTrain);
XTrain = doc2sequence(emb,documentsTruncatedTrain);
for i = 1:numel(XTrain)
    XTrain{i} = leftPad(XTrain{i},sequenceLength);
end
inputSize = embeddingDimension;
outputSize = 180;
numClasses = 5;

layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(512,'OutputMode','last')
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer]
options = trainingOptions('adam', ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.01, ...
    'Plots','training-progress', ...
    'Verbose',0);
net = trainNetwork(XTrain,YTrain,layers,options);
textDataTest = erasePunctuation(textDataTest);
textDataTest = lower(textDataTest);
documentsTest = tokenizedDocument(textDataTest);
documentsTruncatedTest = docfun(@(words) words(1:min(sequenceLength,end)),documentsTest);
XTest = doc2sequence(emb,documentsTruncatedTest);
for i=1:numel(XTest)
    XTest{i} = leftPad(XTest{i},sequenceLength);
end
YPred = classify(net,XTest);
accuracy = sum(YPred == YTest)/numel(YPred)