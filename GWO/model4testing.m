function [ TY ] = model4testing( data4testing, NumberofHiddenNeurons, OutputWeight, par, ActivationFunction )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    dimension = size(data4testing,2);
    NumberofTestingData =  size(data4testing,1);
    weights = par(1,1:NumberofHiddenNeurons*dimension);
    BiasofHiddenNeurons   = par(1, NumberofHiddenNeurons*dimension+1:end)';
    aux = reshape(weights,[dimension, NumberofHiddenNeurons]);
    InputWeight = aux';
    
    TV.P = data4testing';
    tempH_test=InputWeight*TV.P;
    
    ind=ones(1,NumberofTestingData);
    BiasMatrix=BiasofHiddenNeurons(:,ind);             
    
    tempH_test=tempH_test + BiasMatrix;
  switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H_test = 1 ./ (1 + exp(-tempH_test));
    case {'sin','sine'}
        %%%%%%%% Sine
        H_test = sin(tempH_test);        
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H_test = hardlim(tempH_test);        
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H_test = tribas(tempH_test);        
    case {'radbas'}
        %%%%%%%% Radial basis function
        H_test = radbas(tempH_test);        
        %%%%%%%% More activation functions can be added here 
    case {'radbasn'}
        %%%%%%%% Radial basis function
        H_test = radbas(tempH_test);
        %%%%%%%% More activation functions can be added here      
  end
TY=(H_test' * OutputWeight)';                       %   TY: the actual output of the testing data
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
end

