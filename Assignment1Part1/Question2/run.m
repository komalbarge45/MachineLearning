%This is a run file consists of main logic for reading the csv file and
%providing the required inputs to respective functions to train the
%Percetron model for a dataset.
errorInModel = 0;
originalData = readmatrix("TestingData.csv");
data = unique(originalData,'rows'); %Unique values in main dataset.
n_samples = length(data(:,1));
PerceptronWeights=[];
matrix=[];

Perceptron = PerceptronAlgorithm();

for q = 1: 50 %No of iterations in a single epoch
    for c = 1:n_samples
        trainingSample = PerceptronTrainer(data(c,1:3), data(c,4));
        accuracyCheck = Perceptron.trainingModel(trainingSample.inputs, data(c,4), errorInModel);
        if accuracyCheck ~= 0
            errorInModel = accuracyCheck;
        end
    end
    mean_error = (errorInModel*100/n_samples);
    matrix = [matrix;mean_error];
    errorInModel = 0;
    accuracyCheck = 0;
end
%plot a graph for error rate matrix
plot(1:50, matrix, '-');
title("Classification error rate for Perceptron");