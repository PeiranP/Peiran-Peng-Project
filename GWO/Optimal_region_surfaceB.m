function [ X1, Y1, Z1, PC ] = Optimal_region_surfaceB(varX, varY, varZ, X, res, NumberofHiddenNeurons, par, OutputWeight, ActivationFunction )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 ejnor = X;   %raw_data;
 average = mean(ejnor);
 minimo = min(min(ejnor)); % Here I use raw data because I did not split the data set
 maximo = max(max(ejnor)); % when you split/divide the data for cross validation use data for training
 base_vect = average;
 iter = 1;
  y_s = zeros(res,res,res);
%  y_s = zeros(res,res);
 PC = zeros(res,res,res);
%  PC = zeros(res,res);
%..........................................................................
%..........................................................................
for ii = 1:res
    for jj = 1:res
        for ll = 1:res
        % creating input vectors
        %..................................................................
        base_vect(1,varX)= -minimo + jj*(maximo/res);
        base_vect(1,varY)= -minimo + ii*(maximo/res);
        base_vect(1,varZ)= -minimo + ll*(maximo/res);
        x1 = base_vect; % input vector
        iter = iter + 1;
        % Computing the RBF Network output
        %..................................................................
         y_aux = 0;
%..........................................................................
y = 0;
    dimension = size(x1,2);
    NumberofTrainingData =  size(x1,1);
    weights = par(1,1:NumberofHiddenNeurons*dimension);
    BiasofHiddenNeurons   = par(1, NumberofHiddenNeurons*dimension+1:end)';
    aux = reshape(weights,[dimension, NumberofHiddenNeurons]);
    InputWeight = aux';
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
P = x1';
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
     y_aux = (H' * OutputWeight)'; 
%      Z1(jj,ii) = y_aux;
     PC(jj,ii,ll) = y_aux;

        end %
    end %
end
%--------------------------------------------------------------------------
raw_data4training = X;
% create X vector
for jj=1:res
    for ii=1:res % here the conversion from normalised to real data again is based on the fact you normalised between 0-1
%         for ll=1:res
        X1(jj,ii)= min(raw_data4training(:,varX)) + ii*((max(raw_data4training(:,varX)) - min(raw_data4training(:,varX)))/res);
%         end
    end
end
% create Y vector
% for jj=1:res
    for ii=1:res
        for ll=1:res
        Y1(jj,ii)= min(raw_data4training(:,varY)) + ll*((max(raw_data4training(:,varY)) - min(raw_data4training(:,varY)))/res);
        end
    end
% end
% create Z vector
for jj=1:res
%     for ii=1:res
        for ll=1:res
        Z1(jj,ll)= min(raw_data4training(:,varZ)) + jj*((max(raw_data4training(:,varZ)) - min(raw_data4training(:,varZ)))/res);
        end
%     end
end
%+++++++++++++++++++++++++++++++++++++++++++++++++++=++++++++++++++++++++++
end

