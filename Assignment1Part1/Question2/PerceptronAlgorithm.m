classdef PerceptronAlgorithm
    %PerceptronAlgorithm: This class defines the perceptron algorithm
    %related logic where we model the weights and inputs to predict the
    %accurate value.
    
    properties (Constant)
      perceptronWeights = weightsData
    end

    methods
        function weightArray = PerceptronAlgorithm()
        %PerceptronAlgorithm: Constructor returning the perceptron weights matrix
            weightArray.perceptronWeights.weights(1) = 1.6717;
            weightArray.perceptronWeights.weights(2) = -0.6018;
            weightArray.perceptronWeights.weights(3) = -0.9440;
            weightArray.perceptronWeights.weights(4) = 0.3239;
            %Used random weights in first epoch to get maximum accurate weights
            %and changed these weights over to next 50 epochs to come to
            %minimum classification error rate.
            %weightArray.perceptronWeights.weights = rand(1,4);
        end
        
        function outputSum = predictOutput(weightArray, inputs)
        %METHOD1 predictOutput: Calculate the prediction of output using
        %weights and variables
        % weightArray : Weights array
        % inputs : variables(features)
            outputSum = 0;
            for i = 1:length(inputs)
                outputSum = outputSum + weightArray.perceptronWeights.weights(i)*inputs(i);
            end
            if outputSum > 0
                outputSum = 2;
            else
                outputSum = 1;
            end
        end
        
        function accuracyCheck = trainingModel(weightArray,inputs, desiredOutput, ErrorInModel)
        %METHOD2 trainingModel: it trains the model and returns the error
        %occured in the model
        % weightArray : Weights array
        % inputs : variables(features)
        % desiredOutput : Expected output
        % ErrorInModel : Error occurance in the model
        
            predictedValue = weightArray.predictOutput(inputs);
            accuracyCheck = 0;
            modelError = desiredOutput - predictedValue;
            if modelError ~= 0
                for j = 1:length(inputs(1,:))
                    %Learning rate has been varied over the epochs.
                    weightArray.perceptronWeights.weights(j) = weightArray.perceptronWeights.weights(j) + 0.0001 .* modelError .* inputs(j);
                end
                accuracyCheck = ErrorInModel + 1;
            end
        end
    end
end