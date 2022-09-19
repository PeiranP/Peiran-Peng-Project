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
end
%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
C = 65;
OutputWeight = pinv(H') * T';
% Computing Root-Mean-Square-Error (RMSE)
%OutputWeight=inv(eye(size(H,1))/C+H * H') * H * T';   % faster method 1 //refer to 2012 IEEE TSMC-B paper
%implementation; one can set regularizaiton factor C properly in classification applications 
%OutputWeight=(eye(size(H,1))/C+H * H') \ H * T';      % faster method 2 //refer to 2012 IEEE TSMC-B paper
%implementation; one can set regularizaiton factor C properly in classification applications
% Calculating the output
Y=(H' * OutputWeight)'; 
 rmse=sqrt(mse(T - Y));  
%..........................................................................
end

