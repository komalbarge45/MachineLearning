%% Load data for support vector machine algorithm
data = load("trainingData.csv");
y = load("trainingLabels.mat");
y = y.mainOutput;

%% SVM template 
t = templateSVM('Standardize',true);

%% Model for SVM algorithm
Mdl = fitcecoc(data,y,'Learners',t,...
    'ClassNames',[1,2,3],...
    'Verbose',2);
XTest = load('testingData.csv');
YTest = load('testingLabelsSVM.mat');
XTest = XTest(1:5500,:);
YTest = YTest.mainO(1:5500,:);
labels = predict(Mdl,XTest);
idx = randsample(size(XTest,1),10);

%% Table for random 10 true and predicted labels
table(YTest(idx),labels(idx),...
    'VariableNames',{'TrueLabels','PredictedLabels'})

MissClassRate = 0;
for i = 1 : size(labels, 1)
    if labels(i,1)~=YTest(i,1)
        MissClassRate=MissClassRate+1;
    end
end

%% Accuracy calculations for simple SVM algorithm
Accuracy = 1 - MissClassRate/i;
disp("Total model accuracy without feature selection");
disp(Accuracy);

%% Confusion matrix for class labels
figure
confusionMatrix = confusionchart(labels,YTest);
confusionMatrix.Normalization = 'row-normalized';
cm = confusionMatrix.NormalizedValues;
top1AccwithoutFeatureSelection = max(max(cm)) * 100;
disp("Accuracy without feature selection");
disp(top1AccwithoutFeatureSelection);

%% Feature selection with Minimum Redundancy Maximum Relevance (MRMR) Algorithm
[idx,scores] = fscmrmr(data,y);
figure
bar(scores(idx))
xlabel('Predictor rank')
ylabel('Predictor importance score');

%% Model calculations with features selection
Mdl = fitcecoc(data(:,idx(1:280)),y,'Learners',t,...
    'ClassNames',[1,2,3],...
    'Verbose',2);

%% Predict labels for testing sample with 280 top ranked features
XTest = XTest(:,idx(1:280));
labels = predict(Mdl, XTest);
idsample = randsample(size(XTest,1),10);

%% Table for random 10 true and predicted labels with feature extraction
table(YTest(idsample),labels(idsample),...
    'VariableNames',{'TrueLabels','PredictedLabels'})

MissClassRate = 0;
for i = 1 : size(labels, 1)
    if labels(i,1)~=YTest(i,1)
        MissClassRate=MissClassRate+1;
    end
end

%% Accuracy calculations for simple SVM algorithm with feature extraction
Accuracy = 1 - MissClassRate/i;
disp("Total model accuracy without feature selection");
disp(Accuracy);

%% Confusion matrix for class labels
figure
confusionMatrix = confusionchart(labels,YTest);
confusionMatrix.Normalization = 'row-normalized';
cm = confusionMatrix.NormalizedValues;
top1AccwithFeatureSelection = max(max(cm)) * 100;
disp("Accuracy with feature selection");
disp(top1AccwithFeatureSelection);
