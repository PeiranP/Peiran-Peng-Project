%  Simple Continuous GA
% minimizes the objective function designated in ff
% Before beginning, set all the parameters in parts
% I, II, and III
% Haupt & Haupt
% Adrian Rubio-Solis
% Institute for Materials Discovery (UCL)
% 2003
%__________________________________________________________________________
% Reading the Data Set
clear;
clc;
load processed_data.mat; % reading the data set
X = normalisation_1( X, 2 );
%--------------------------------------------------------------------------
%                       CROSS-VALIDATION DATA SPLIT
% Spliting the data into two subsets: Training and testing
%--------------------------------------------------------------------------
samples4training = round(0.6*size(X,1));
dimension = size(X,2);
samples4testing = size(X,1) - (samples4training);

px = randperm(size(X,1));
% Variables for training
data4training = zeros(samples4training,dimension);
label4training = zeros(samples4training,1);
% Variables for testing
data4testing = zeros(samples4testing,dimension);
label4testing = zeros(samples4testing,1);
%..........................................................................
%..........................................................................
%%                           Data for Training
%..........................................................................
index = 0;
PROPERTY_STEEL = 2; % PROPERTY_STEEL = 1= Yield Strength,PROPERTY_STEEL = 2 = Tensile strength
                    % PROPERTY_STEEL = 3 = Elasticity Strength
   for k = 1:samples4training
         index = index + 1;
         data4training(index,:) = X(px(k),1:end);
         label4training(index,1) = Y(px(k),PROPERTY_STEEL);
   end
%           Creating Testing Data
index = 0;
%..........................................................................
    for k = samples4training + 1:samples4training + samples4testing 
        index = index + 1;
                data4testing(index,:) = X(px(k),1:end);
                label4testing(index,1) = Y(px(k),PROPERTY_STEEL);
    end
%..........................................................................
%__________________________________________________________________________
% I Setup the GA  Parameters
NumberofHiddenNeurons = 100;
npar = NumberofHiddenNeurons*dimension + NumberofHiddenNeurons; % number of optimization variables 
varhi=-1; % high limit
varlo=1; % lower limit
%__________________________________________________________________________
maxit = 10;                      % Max number of iterations or max number of generations
best_fitness = zeros(maxit,1);
mincost=-9999999;
                            %II Stopping criteria
                   % max number of iterations % minimum cost
%__________________________________________________________________________
%                           III GA parameters 
popsize = 10;   % set population size   (only pair numbers) 
mutrate = 0.09;   % 10% the reference set mutation rate
selection = 0.55; % 450% or the number of individuals that survivemore than one generation
%                fraction of population kept
Nt = npar; % continuous parameter GA Nt=#variables

keep = floor(selection*popsize); % #population (number of individuals)
                                 % number of members that survive
                                 % members that survive
nmut = ceil((popsize-1)*Nt*mutrate); % total number of % mutations
M = ceil((popsize-keep)/2);          % number of matings
%__________________________________________________________________________ 
%                   Create the initial population
%__________________________________________________________________________
generation = 0; % generation counter initialized
par=(varhi-varlo)*rand(popsize,npar)+varlo;     % random generation of parameters
% evaluating initial cost (fitness) of the initial population
InputWeight = zeros(NumberofHiddenNeurons,dimension);
NumberofTrainingData = samples4training;
ActivationFunction = 'radbas';        % Select the type of hidden neuron model
% List of models for the hidden neurons
% sine, hardlim, tribas, radbas
%__________________________________________________________________________
for i = 1:popsize  
    % Convert chromosome into a matrix format
    % par is the variable used to define the crhomosone of each individual
    [ c(i).OutputWeight, Y, rmse ] = ELM_model( par(i,:), NumberofHiddenNeurons, data4training, label4training,ActivationFunction);
    cost(i,1) = rmse;
end                    
[cost,ind] = sort(cost);          % min cost in element 1
par = par(ind,:);               % sort continuous
minc(1) = min(cost);            % minc contains min of
meanc(1)= mean(cost);           % meanc contains mean of population
%__________________________________________________________________________ 
                        % Iterate through generations
rmse_training = zeros(maxit,1);   
% Code added recently (27-07-2021)
T = label4training';
R2_training = zeros(maxit,1);

%  HERE ids where the loop for iterating through generations STARTS                        
while generation<maxit
 
 generation = generation+1;             % increments generation counter
%.......................................................................... 
% %                      - Pair and mate - 
%..........................................................................
 M = ceil((popsize-keep)/2);                    % number of matings
 % Roulette WHEEL
 prob = flipud([1:keep]'/sum([1:keep]));        % weights
%                                               % chromosomes
 odds=[0 cumsum(prob(1:keep))'];                % probability      
%                                               % probability
%                                               % distribution
 pick1=rand(1,M);                               %  mate #1
 pick2=rand(1,M);                               % mate #2
% % ma and pa contain the indicies of the chromosomes 
% % that will mate
 ic=1;
 % loop for the number of matings
 while ic<=M
  % Pick any two individuals based on the probability of the roulette wheel
  for id=2:keep+1
     if pick1(ic)<=odds(id) & pick1(ic)>odds(id-1)
         ma(ic)=id-1;
     end
     if pick2(ic)<=odds(id) & pick2(ic)>odds(id-1) 
         pa(ic)=id-1;
     end
   end
    ic=ic+1; 
 end
%__________________________________________________________________________
%..........................................................................
%    - Performs mating using single point crossover -   MATING
%..........................................................................
 ix = 1:2:keep;                            % index of mate 1
 xp = ceil(rand(1,M)*Nt);                  % cross over point 
 r = rand(1,M);                            % mixing parameter
 for ic=1:M
     xy = par( ma(ic),xp(ic) ) - par(pa(ic),xp(ic)); % ma and pa    
%                                               % mate
     par(keep+ix(ic),:)   =   par(ma(ic),:); % 1st offspring 
     par(keep+ix(ic)+1,:) =   par(pa(ic),:); % 2nd offspring 
% % % 1st offspring
  par(keep+ix(ic),xp(ic))= par(ma(ic),xp(ic)) - r(ic).*xy;
% % % 2nd offspring
  par(keep+ix(ic)+1,xp(ic))=par( pa(ic),xp(ic) ) + r(ic).*xy; 
  if xp(ic)<npar % crossover when last variable not selected
      par(keep+ix(ic),:)= [par(keep+ix(ic),1:xp(ic)) par(keep+ix(ic)+1,xp(ic)+1:npar)];
      par(keep+ix(ic)+1,:)= [par(keep+ix(ic)+1,1:xp(ic)) par(keep+ix(ic),xp(ic)+1:npar)];
  
   end % if 
 end
%__________________________________________________________________________ 
                           % Mutate the population 
 mrow=sort(ceil(rand(1,nmut)*(popsize-1))+1); 
 mcol=ceil(rand(1,nmut)*Nt);
  
  for ii=1:nmut 
      par(mrow(ii),mcol(ii))=(varhi-varlo)*rand+varlo;
                              % mutation 
  end % ii
%__________________________________________________________________________
% The new offspring and mutated chromosomes are % evaluated
%cost=feval(ff,par);
for i = 1:popsize
    %cost(i,1) = feval(ff, par(i,:));   
    % Convert chromosome into a matrix format
    [ c(i).OutputWeight,Y, rmse ] = ELM_model( par(i,:), NumberofHiddenNeurons, data4training, label4training,ActivationFunction);
    cost(i,1) = rmse;
end
%__________________________________________________________________________ 
 % Sort the costs and associated parameters 
 [cost,ind]=sort(cost);
 par=par(ind,:);
%__________________________________________________________________________ 
% Do statistics for a single nonaveraging run
    minc(generation+1) = min(cost);
    meanc(generation+1)= mean(cost);
%__________________________________________________________________________ 
                              % Stopping criteria
   if generation > maxit | cost(1)<mincost
       best_fitness(generation,1)=cost(1);
       break 
   else
       best_fitness(generation,1)=cost(1);
   end
   rmse_training(generation,1) = best_fitness(generation,1);
   % HERE e recorded (saved) the R2 for every/ach eolution
   R2_training(generation,1) = rsquare(T',Y');
   [generation cost(1)  R2_training(generation,1)] 
   
 end %number_of_generation
%__________________________________________________________________________ 
%%                       - Testing Results/stage-
%__________________________________________________________________________ 
% In this stage, the parameters und for the perceptron model are tested
% withew/unseen data created by spliting the originaata Such data is
% usually called testing data
%__________________________________________________________________________
% caklculate matrx H for testing data
    %
  [ TY ] = model4testing( data4testing, NumberofHiddenNeurons, c(1).OutputWeight, par(ind(1),1:end), ActivationFunction );
% Computing Root-Mean-Square-Error (RMSE) for testing
  TV.T = label4testing';
  rmse_test=sqrt(mse(TV.T - TY));  
%..........................................................................
%                  - SECTION FOR PLOTTING THE 3D SURFACE -
%..........................................................................
 %%
 varX = 15;
 varY = 26;
 res = 50;
[ X1, Y1, Z1 ] = Optimal_region_surface( varX, varY, X, res, NumberofHiddenNeurons, par(ind(1),1:end), c(1).OutputWeight, ActivationFunction );
%--------------------------------------------------------------------------
% Plotting Surface
surf(X1,Y1,Z1,'FaceAlpha',0.5);


%% Section to calculate the correlation coefficient for training and testing
% Update
% for testing
 R2_testing = rsquare(TV.T',TY');
% 
 R2_training_final = rsquare(T',Y');
 [R2_training_final R2_testing]
%%
 mae_final = mae((T-Y));
 mae_final = mae((T-Y));
 
 
 
 
 
 
