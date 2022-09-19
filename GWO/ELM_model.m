function [ OutputWeight, Y, rmse ] = ELM_model( chromosome, NumberofHiddenNeurons, data4training, label4training, ActivationFunction )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    dimension = size(data4training,2);
    NumberofTrainingData =  size(data4training,1);
    weights = chromosome(1,1:NumberofHiddenNeurons*dimension);
    BiasofHiddenNeurons   = chromosome(1, NumberofHiddenNeurons*dimension+1:end)';
    aux = reshape(weights,[dimension, NumberofHiddenNeurons]);
    InputWeight = aux';
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
P = data4training';
T = label4training';
%..........................................................................
tempH=InputWeight*P;
clear P;                                            %   Release input of training data 
ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH=tempH+BiasMatrix;

%%%%%%%%%%% Calculate hidden neuron output matrix H
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H = 1 ./ (1 + exp(-tempH));
    case {'sin','sine'}
        %%%%%%%% Sine
        H = sin(tempH);    
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H = double(hardlim(tempH));
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H = tribas(tempH);
    case {'radbas'}
        %%%%%%%% Radial basis function
        H = radbas(tempH);
        %%%%%%%% More activation functions can be added here    
    case {'hyper'}
        %%%%%%%% Hyperbolic tangent function
        H = (1-exp(-tempH))/(1+exp(-tempH));
%     case {'relu'}
%         for o=1:size(tempH,1)
%             if tempH(o,1)>0
%                 H(o,1)=tempH(o,1);
%             else
%                 H(o,1)=0;
%             end
%         end
end
%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
OutputWeight = pinv(H') * T';
% Calculating the output
Y=(H' * OutputWeight)'; 
% Computing Root-Mean-Square-Error (RMSE)
 rmse=sqrt(mse(T - Y));  
%..........................................................................
end

