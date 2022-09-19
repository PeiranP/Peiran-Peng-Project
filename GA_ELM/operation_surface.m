function [ X, Y, Z1 ] = operation_surface( varX, varY, ejnor, no_of_grid )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%..........................................................................
%                  - SECTION FOR PLOTTING THE 3D SURFACE -
%..........................................................................
%%
%varX = 11;
%varY = 17;
%Ci = C; wj = w;
%[ ejnor ] = data_normalisation01(raw_data); % Added by Ali on 10/11/17
%ejnor = x;%raw_data;
%[ncl, dim] = size(Ci);
average = mean(ejnor);
res = no_of_grid; % the scale for griding
minimo = min(min(ejnor)); % Here I use raw data because I did not split the data set
maximo = max(max(ejnor)); % when you split/divide the data for cross validation use data for training
base_vect = average;
% varX = 1;  % first variable to plot surface
% varY = 3;
iter = 1;
y_s = zeros(res,res);
Z = zeros(res,res);
Z1 = zeros(res,res);
%..........................................................................
%..........................................................................
for ii = 1:res
    for jj = 1:res
        % creating input vectors
        %..................................................................
        base_vect(1,varX)= -minimo + jj*(maximo/res);
        base_vect(1,varY)= -minimo + ii*(maximo/res);
        x1 = base_vect; % input vector
        iter = iter + 1;
        % Computing the RBF Network output
        %..................................................................
         y_aux = 0;
     %    gm = zeros(1,ncl);
        %..................................................................
%..........................................................................
y = 0;
%gm = zeros(1,ncl);
%..........................................................................
%    RBFNN Structure, section to include your model. ENSEMBLE of Mamdani
%    Type
%..........................................................................
for model = 1:Number_of_networks
   net(model).y(iter,1) = 0.0;
   %[net(model).Mr]
 [ net(model).y(iter), net(model).gm ] =...
     rbf_Boot( net(model).Mr, dim, x1, net(model).ci, net(model).s, net(model).wj );
y_aux = y_aux + net(model).y(iter); % Network Output
end
%..........................................................................
y_aux = y_aux/Number_of_networks; % Ensemble Output
%..........................................................................
%..........................................................................
        %..................................................................
        y_s(jj,ii) = y_aux;
        %Z(jj,ii) = ( y_s(jj,ii)*( max(labels_base(:)) - min(labels_base(:)) ) ) + min(labels_base(:)); % if output is normalised use the commented text left to this!
        Z(jj,ii) = y_aux;%( y_s(jj,ii) - min(labels_base(:)) )/  ( max(labels_base(:)) - min(labels_base(:)) ) ;
        [y_aux Z(jj,ii)];
        %}
        %..................................................................
    end %
end %
%--------------------------------------------------------------------------
maximo = max(max(Z));
minimo = min(min(Z));
for ii = 1:res
    for jj = 1:res
       Z1(jj,ii) = ( Z(jj,ii) - minimo )/  ( maximo - minimo ) ; 
    end
end
%--------------------------------------------------------------------------
%..........................................................................
% create X vector
for jj=1:res
    for ii=1:res % here the conversion from normalised to real data again is based on the fact you normalised between 0-1
        X(jj,ii)= min(raw_data4training(:,varX)) + ii*((max(raw_data4training(:,varX)) - min(raw_data4training(:,varX)))/res);
    end
end
% create Y vector
for jj=1:res
    for ii=1:res
        Y(ii,jj)= min(raw_data4training(:,varY)) + ii*((max(raw_data4training(:,varY)) - min(raw_data4training(:,varY)))/res);
    end
end

end

