classdef PerceptronTrainer
    %PerceptronTrainer: This class defines the feature variables related
    %logic. It consists of a constructor of input variables in the dataset.

    properties
        inputs
        answer
    end

    methods
        function obj = PerceptronTrainer(x,desiredAns)
        %PerceptronTrainer: Constructor returning an object matrix of input
        %variables and expected output
        
        obj.inputs(1) = x(1,1);
        obj.inputs(2) = x(1,2);
        obj.inputs(3) = x(1,3);
        obj.inputs(4) = 1;   %bias input
        obj.answer = desiredAns;
        end
    end
end
