function [TrainingTime] = assignment_elm(trainingData,testingData, elm_type)
%% assignment_elm is a function for I-ELM algorithm, 
% where trainingData is a training dataset,
% testingData is a testing dataset,
% elm_type is whether passed dataset is for regression or classification
% if it's a classification data, pass elm_type=0, and for regression pass elm_type=1

%Training dataset 
trainingData=readmatrix(trainingData);
labels_TrainingData=trainingData(1:10700,1)';
mainData=trainingData(1:10700,2:size(trainingData,2))';
clear trainingData;
mainData = double(mainData);
labels_TrainingData = double(labels_TrainingData);

%Testing dataset
testingData = readmatrix(testingData);
labels_testingData = testingData(1:8500,1)';
testData=testingData(1:8500,2:size(testingData,2))';
clear testingData;
testData = double(testData);
labels_testingData = double(labels_testingData);

NumberofNeurons = 1;
NumberofInputNeurons=size(mainData,1);
start_time_train=cputime;
eta = 1;
Error = labels_TrainingData;
E = sqrt(sum(Error .* Error));
L = 2;
NeuronNodemax = 10;
C=2;

% ELM on training dataset
InputWeight(:,:,1)=rand(NumberofNeurons,NumberofInputNeurons)*0.2-0.1;
temporaryH=InputWeight*mainData*-0.00001;
H(:,:,1) = 1 ./ (1 + exp(-temporaryH));
betaWeight(:,:,1)=pinv(H(:,:,1)') .* labels_TrainingData;

while(L <= NeuronNodemax & E > eta)
  InputWeight(:,:,L)=rand(NumberofNeurons,NumberofInputNeurons)*0.2-0.1;
  temporaryH=InputWeight(:,:,L)*mainData*-0.00001; % tempH = aX - b
  H(:,:,L) = 1 ./ (1 + exp(-temporaryH)); %activation function
  betaWeight(:,:,L)=pinv(H(:,:,L)') .* Error; %betaW = inv(H).E
  Error = Error - betaWeight(:,:,L) .* H(:,:,L);  % E = T-betaW.H
  labels_TrainingData = Error*0.1;
  L = L + 1;
end

end_time_train=cputime;
TrainingTime=end_time_train-start_time_train;
disp("Training time = ");
disp(TrainingTime);
yTrain= H(:,:,NeuronNodemax) .* betaWeight(:,:,NeuronNodemax)*1000;

% ELM on Testing dataset
start_time_test=cputime;
for neuron = 1:NeuronNodemax
  temporaryTestH=InputWeight(:,:,neuron)*testData*-0.00001;
  Htest(:,:,neuron) = 1 ./ (1 + exp(-temporaryTestH));
end

end_time_test=cputime;
yfinal = labels_testingData;
TestingTime=end_time_test-start_time_test;
disp("Testing time = ");
disp(TestingTime);
for n= 1:NeuronNodemax
    yfinal = yfinal + Htest(:,:,n).*betaWeight(:,1:size(labels_testingData, 2),n)*10;
end

%%Accuracy for classification
if elm_type == 0
    MissClassRate=0;
    for i = 1 : size(labels_TrainingData, 2)
        label_expected=round(labels_TrainingData(:,i));
        label_actual=round(max(yTrain(:,i)));
        if label_actual~=label_expected
            MissClassRate=MissClassRate+1;
        end
    end
    %%Training accuracy(RMSE) for classification
    TrainingAccuracy=1-MissClassRate/size(labels_TrainingData,2);
    MissClassTestRate=0;
    
    for i = 1 : size(labels_testingData, 2)
        label_test_expected=labels_testingData(:,i);
        label_test_actual=max(yfinal(:,i));
        if label_test_actual~=label_test_expected
            MissClassTestRate=MissClassTestRate+1;
        end
    end
    %%Testing accuracy(RMSE) for classification
    TestingAccuracy=1-MissClassTestRate/size(labels_testingData,2);
end

%Accuracy for regression
if elm_type == 1
    %%Testing and training accuracy(RMSE) for regression
    TrainingAccuracy=sqrt(mse(labels_TrainingData - yTrain));
    TestingAccuracy=sqrt(mse(labels_testingData - yfinal)) ;
end

disp("Training accuracy = ");
disp(TrainingAccuracy);
disp("Testing accuracy = ");
disp(TestingAccuracy);