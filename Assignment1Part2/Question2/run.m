%This is a run file consists of main logic for reading the csv file and
%providing the required inputs to respective functions to train the
%model with newton's algorithm for a dataset.
errorInModel = 0;
originalData = readmatrix("TestingData.csv");
data = unique(originalData,'rows'); %Unique values in main dataset.
n_samples = length(data(:,1));
PerceptronWeights=[];
accuracyMatrix=[];
ee=[];
mse=[];
errorRate = 0;
Perceptron = PerceptronAlgorithm();
n_epoch = 30;
for q = 1: n_epoch %No of epochs
    for c = 1:n_samples
        trainingSample = PerceptronTrainer(data(c,1:3), data(c,4));
        [errorRate, accuracyCheck] = Perceptron.trainingModel(trainingSample.inputs, data(c,4), errorRate);
        ee = [ee;accuracyCheck(1,1)];
    end
    disp(Perceptron.perceptronWeights.weights);
    mean_error = 100-(errorRate*100/n_samples);
    mse(q) = mean(ee.^2);
    accuracyMatrix = [accuracyMatrix;mean_error];
    errorRate = 0;
    %errorInModel = 0;
    accuracyCheck = 0;
end
%plot a graph for error rate matrix
figure;
plot(1:n_epoch, mse, '-');
title("MSE for Newton's Method");
xlabel("No of epochs");
ylabel("MSE");
%plot a graph for testing accuracy
figure;
hold on;
plot(1:n_epoch, accuracyMatrix, '-');
title("Accuracy for Newton's Method");
xlabel("No of epochs");
ylabel("Accuracy in %");