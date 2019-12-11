%% Fetch the images from Caltech256 dataset
imageData = imageDatastore('Caltech256\256_ObjectCategories',...
'LabelSource', 'foldernames', 'IncludeSubfolders', true);

%% GPU device used for this program
disp('GPU device used for this program');
gpuDevice(1)

%% Split the labels 
% Use 30 images for training and remaining images for testing purpose
disp('1. Spliting labels for training and testing');
[trainingSet, testingSet] = splitEachLabel(imageData, 30);

%% Load Resnet101
disp('2. Loading Pretrained Resnet101');
net = resnet101;

%% Redefining read function to process images
disp('3. Preprocessing images for Resnet101 network');
trainingSet.ReadFcn = @(filename)PrepImageWithResnet101Dim(filename);
testingSet.ReadFcn = @(filename)PrepImageWithResnet101Dim(filename);

 %% Get features from Resnet101
disp('4. Extracting Resnet101 deeper layer features');
extractionLayer = 'fc1000';
resnet101_trainFeatures = activations(net,trainingSet,extractionLayer,'MiniBatchSize',120);
resnet101_trainFeatures = reshape(resnet101_trainFeatures,[1*1*1000,size(resnet101_trainFeatures,4)])' ;
resnet101_testFeatures = activations(net,testingSet,extractionLayer,'MiniBatchSize',120);
resnet101_testFeatures = reshape(resnet101_testFeatures,[1*1*1000,size(resnet101_testFeatures,4)])';
 
%% Load xception neural network
disp('5. Loading xception neural network');
net = xception;

%% Redefining read function to process images
disp('6. Preprocessing images for xception network');
trainingSet.ReadFcn = @(filename)PrepImageWithXceptionDim(filename);
testingSet.ReadFcn = @(filename)PrepImageWithXceptionDim(filename); 

%% Get training set deep features from xception
disp('7. Extracting xception features');
xception_trainFeatures = activations(net,trainingSet,'avg_pool','MiniBatchSize',120);
xception_trainFeatures = reshape(xception_trainFeatures,[1*1*2048,size(xception_trainFeatures,4)])';
xception_testFeatures = activations(net,testingSet,'avg_pool','MiniBatchSize',120);
xception_testFeatures = reshape(xception_testFeatures,[1*1*2048,size(xception_testFeatures,4)])';

%% Merge Resnet and xception deep features for training and testing
disp('8. Merging the features from Resnet101 and xception network');
Mergedtrain = horzcat(xception_trainFeatures, resnet101_trainFeatures);
Mergedtest = horzcat(xception_testFeatures, resnet101_testFeatures);
train_labels = grp2idx(trainingSet.Labels);
test_labels = grp2idx(testingSet.Labels);

%% Creating training and testing dataset
training = horzcat(train_labels,Mergedtrain);
testing = horzcat(test_labels,Mergedtest);

disp('9. Classification using ELM algorithm');
[TrainingTime, TestingAccuracy,Training,Testing] = ELM(training, testing, 1, 10000, 'sig');