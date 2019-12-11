classdef PerceptronAlgorithm
    %PerceptronAlgorithm: This class defines the Newton's optimization algorithm
    %related logic where we model the weights and inputs to predict the
    %accurate value.
    
    properties (Constant)
      perceptronWeights = weightsData
    end

    methods
        function weightArray = PerceptronAlgorithm()
        %PerceptronAlgorithm: Constructor returning the perceptron weights matrix
	    weightArray.perceptronWeights.weights(1) = -0.0337;
            weightArray.perceptronWeights.weights(2) = 0.0318;
            weightArray.perceptronWeights.weights(3) = -0.0638;
            weightArray.perceptronWeights.weights(4) = 2.5511;
            %Used random weights in first epoch to get maximum accurate weights
            %and changed these weights over to next 30 epochs to come to
            %minimum classification error rate.
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
        
        function [errorRate, accuracyCheck] = trainingModel(weightArray,inputs, desiredOutput, errorRate)
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
                errorRate = errorRate + 1;
%               Commented code is for LMS algorithm
%               for j = 1:length(inputs(1,:))
%                   %Learning rate has been varied over the epochs.
%                   errorval =  desiredOutput - weightArray.perceptronWeights.weights(j) * inputs(j);
%                   weightArray.perceptronWeights.weights(j) = weightArray.perceptronWeights.weights(j) + 0.001 * errorval * inputs(j);
%               end
%               //for perceptron - accuracyCheck = ErrorInModel + 1;
%               accuracyCheck = errorval;


%               For Newton's method, 
%               Cosidered optimization weights equation as x1w1+x2w2^4+x3w3^3+x4w4^2 = 0, whereas, 
%               g = diff(w), H = diff^2(w)
%               delta(w) = H^-1 * g
%               w(n+1) = w(n) - delta(w)
                g = 3.* inputs(3) .* weightArray.perceptronWeights.weights .^2 + 4 .* inputs(2) .* weightArray.perceptronWeights.weights .^3 + 2.* inputs(4) .* weightArray.perceptronWeights.weights + inputs(1);
                H = 6.* inputs(3) .* weightArray.perceptronWeights.weights + 12.* inputs(2) .* weightArray.perceptronWeights.weights .^2 + 2.*inputs(4);
                deltaW = pinv(H) .* g;
                deltaW = reshape(deltaW',1,[]);
                weightArray.perceptronWeights.weights = weightArray.perceptronWeights.weights - deltaW(1:4);
                accuracyCheck = deltaW(4);
            end
        end
    end
end