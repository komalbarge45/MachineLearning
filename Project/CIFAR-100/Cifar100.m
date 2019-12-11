%% Load Googlenet pretrained neural network
network = googlenet;
layers = network.Layers;
lgraph = layerGraph(network);

%% Replace last 3 layers in GoogLeNet network
%% Number of classes provided to fully connected layer is 100
lgraph = replaceLayer(lgraph,'loss3-classifier', fullyConnectedLayer(100,'Name','fcLayer','WeightLearnRateFactor',20,'BiasLearnRateFactor', 20));
lgraph = replaceLayer(lgraph,'prob',softmaxLayer('Name','smLayer'));
lgraph = replaceLayer(lgraph,'output',classificationLayer('Name','outputLayer'));
 
%% Load images for CIFAR100 dataset
Ximds = imageDatastore('CIFAR-100\TRAIN\','IncludeSubfolders',true,'LabelSource','foldernames');
Xtestimds = imageDatastore('CIFAR-100\TEST\','IncludeSubfolders',true,'LabelSource','foldernames');

%% Data augmentation for images to fit into googlenet
X = augmentedImageDatastore([224 224],Ximds);
Xtest = augmentedImageDatastore([224 224],Xtestimds);

%% Options to be provided for network
miniBatchSize = 30;
opts = trainingOptions('sgdm',...
    'MiniBatchSize',miniBatchSize,... %set mini batch size
      'LearnRateSchedule','piecewise',...
      'LearnRateDropFactor',0.1,... 
      'LearnRateDropPeriod',3,... 
      'MaxEpochs',10,...
      'InitialLearnRate',1e-3,...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress',...
    'ExecutionEnvironment','auto');
start = cputime;
 
%% Train network
network = trainNetwork(X, lgraph, opts);
endtime=cputime;
predictedlabels = classify(network, Xtest);
 
%% Accuracy for pre-trained network model
accuracy = mean(predictedlabels == Xtestimds.Labels);
fprintf('Accuracy for pretrained GoogLeNet network for CIFAR-100: %s \n', accuracy);
fprintf('Time required for training a network: %s \n', endtime-start);