%  Simple Continuous GA
% minimizes the objective function designated in ff
% Before beginning, set all the parameters in parts
% I, II, and III
% Haupt & Haupt
% 2003
%_______________________________________________________
% I Setup the GA
ff= 'f10'; % objective function
npar=900; % number of optimization variables 
varhi=4; varlo=-4; % variable limits

%_______________________________________________________
%
maxit=1000;
best_fitness = zeros(maxit,1);
mincost=-9999999;
%II Stopping criteria
% max number of iterations % minimum cost
%__________________________________________________________________________
%                           III GA parameters 
popsize=12;   % set population size 
mutrate=0.1;   % set mutation rate
selection=0.5; % fraction of population kept
Nt = npar; % continuous parameter GA Nt=#variables

keep = floor(selection*popsize); % #population (number of individuals)
                               % members that survive
nmut = ceil((popsize-1)*Nt*mutrate); % total number of % mutations
M = ceil((popsize-keep)/2);          % number of matings
%__________________________________________________________________________ 
%            Create the initial population
number_of_generation=0; % generation counter initialized

par=(varhi-varlo)*rand(popsize,npar)+varlo;     % random generation of parameters

% evaluating initial cost (fitness) of the initial population
for i = 1:popsize
    cost(i,1) = feval(ff, par(i,:));   
end
%feval(ff,par);       % calculates population cost
                                % using ff                        
[cost,ind] = sort(cost);          % min cost in element 1
par = par(ind,:);               % sort continuous
minc(1) = min(cost);            % minc contains min of
meanc(1)= mean(cost);           % meanc contains mean of population
%__________________________________________________________________________ 
% Iterate through generations
while number_of_generation<maxit
 
 number_of_generation = number_of_generation+1;                      % increments generation counter
% %_______________________________________________________ 
% %                      Pair and mate 
 M = ceil((popsize-keep)/2);                    % number of matings
 prob = flipud([1:keep]'/sum([1:keep]));        % weights
%                                              % chromosomes
 odds=[0 cumsum(prob(1:keep))'];              % probability      
%                                              % probability
%                                              % distribution
 pick1=rand(1,M);                             % mate #1
 pick2=rand(1,M);                             % mate #2
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
% 
% %__________________________________________________________________________
% 
% % % Performs mating using single point crossover
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
    cost(i,1) = feval(ff,par(i,:));   
end
%__________________________________________________________________________ 
 % Sort the costs and associated parameters 
 [cost,ind]=sort(cost);
 par=par(ind,:);
%__________________________________________________________________________ 
% Do statistics for a single nonaveraging run
    minc(number_of_generation+1) = min(cost);
    meanc(number_of_generation+1)= mean(cost);
%__________________________________________________________________________ 
                              % Stopping criteria
   if number_of_generation > maxit | cost(1)<mincost
       best_fitness(number_of_generation,1)=cost(1);
       break 
   else
       best_fitness(number_of_generation,1)=cost(1);
   end
   [number_of_generation cost(1)] 
   
 end %number_of_generation
%__________________________________________________________________________ 
%                           Displays the output
%__________________________________________________________________________
