%___________________________________________________________________%
%  Grey Wolf Optimizer (GWO) source codes version 1.0               %
%                                                                   %
%  Developed in MATLAB R2011b(7.13)                                 %
%                                                                   %
%  Author and programmer: Seyedali Mirjalili                        %
%                                                                   %
%         e-Mail: ali.mirjalili@gmail.com                           %
%                 seyedali.mirjalili@griffithuni.edu.au             %
%                                                                   %
%       Homepage: http://www.alimirjalili.com                       %
%                                                                   %
%   Main paper: S. Mirjalili, S. M. Mirjalili, A. Lewis             %
%               Grey Wolf Optimizer, Advances in Engineering        %
%               Software , in press,                                %
%               DOI: 10.1016/j.advengsoft.2013.12.007               %
%                                                                   %
%___________________________________________________________________%
% Modified by Adrian Rubio Solis, UCL - Institute for Materials Discovery
% Implemented together with Extreme Learning Machine for Perceptron Neural
% Networks
% You can simply define your cost in a seperate file and load its handle to fobj 
% The initial parameters that you need are:
%__________________________________________
% fobj = @YourCostFunction
% dim = number of your variables
% Max_iteration = maximum number of generations
% SearchAgents_no = number of search agents
% lb=[lb1,lb2,...,lbn] where lbn is the lower bound of variable n
% ub=[ub1,ub2,...,ubn] where ubn is the upper bound of variable n
% If all the variables have equal lower bound you can just
% define lb and ub as two single number numbers

% To run GWO: [Best_score,Best_pos,GWO_cg_curve]=GWO(SearchAgents_no,Max_iteration,lb,ub,dim,fobj)
%__________________________________________

%%
tic
clear;
clc;
% load processed_data.mat; % reading the data set
load CIGS_solarC.mat
% load CIGS_solar - copy original.mat %dataset with no missing parameters
X = Paratwo(:,1:14);
Y = Paratwo(:,15:18);
X_nonN = Paratwo;

Xcopy = X;
max_values = max(Xcopy);
min_values = min(Xcopy);

X = normalisation_1( X, 2 );

%--------------------------------------------------------------------------
%                       CROSS-VALIDATION DATA SPLIT
% Spliting the data into two subsets: Training and testing
%--------------------------------------------------------------------------
samples4training = round(0.8*size(X,1));
dimension = size(X,2);
samples4testing = size(X,1) - (samples4training);% here the number of samples for testing is computed

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

 Property_cells= 4; % 1-short circuit current(Jsc) 2-opencircuit voltage(Voc) 3-the fill factor(FF) 4-photoelectric conversion efficiency(PCE)
   for k = 1:samples4training
         index = index + 1;
%          data4training(index,:) = X(px(k),1:end);
%          label4training(index,1) = Y(px(k),PROPERTY_STEEL);
         data4training(index,:) = X(px(k),1:end);
         label4training(index,1) = Y(px(k),Property_cells);
   end
%           Creating Testing Data
index = 0;
%..........................................................................
    for k = samples4training + 1:samples4training + samples4testing 
        index = index + 1;
%               data4testing(index,:) = X(px(k),1:end);
%               label4testing(index,1) = Y(px(k),PROPERTY_STEEL);
                data4testing(index,:) = X(px(k),1:end);
                label4testing(index,1) = Y(px(k),Property_cells);
    end
%..........................................................................
%% Parameters for the Perceptron Neural Network
NumberofHiddenNeurons = 8; % no more than 300
ActivationFunction = 'sine';
T = label4training';
% List of models for the hidden neurons
% sine, hardlim, tribas, radbas
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Parameters for Grey Wolf Optimization
SearchAgents_no=42; % Number of search agents
%Function_name='F10'; % Name of the test function that can be from F1 to F23 (Table 1,2,3 in the paper)
Max_iter=350; % Maximum numbef of iterations
% Load details of the selected benchmark function
%[lb,ub,dim,fobj]=Get_Functions_details(Function_name);
dim = NumberofHiddenNeurons*dimension + NumberofHiddenNeurons; % number of parameters (number of input weights + number of biases)  
lb = -3; % lb = lower boundary min values
ub = 3; % ub = upper boundary max value for the parameters in the NN
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%[Best_score,Best_pos,GWO_cg_curve]=GWO(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
% the function commented above should be expanded below in order to
% implement GWO for ELM (Extreme Learning Machine)
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% initialize alpha, beta, and delta_pos
Alpha_pos=zeros(1,dim);
Alpha_score=inf; %change this to -inf for maximization problems

Beta_pos=zeros(1,dim);
Beta_score=inf; %change this to -inf for maximization problems

Delta_pos=zeros(1,dim);
Delta_score=inf; %change this to -inf for maximization problems

%Initialize the positions of search agents
Positions=initialization(SearchAgents_no,dim,ub,lb);

Convergence_curve=zeros(1,Max_iter);

l=0;% Loop counter
% RMSE for training
rmse_training = zeros(Max_iter,1);
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%                              - Main loop -
while l<Max_iter
    for i=1:size(Positions,1)  
        
       % Return back the search agents that go beyond the boundaries of the search space
        Flag4ub=Positions(i,:)>ub;
        Flag4lb=Positions(i,:)<lb;
        Positions(i,:)=(Positions(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;               
        
        % Calculate objective function for each search agent
        %fitness=fobj(Positions(i,:)); % fitness_function in the original GWO 
        [ c(i).OutputWeight, Y, fitness ] = ELM_model( Positions(i,:), NumberofHiddenNeurons, data4training, label4training,ActivationFunction);
        
        % Update Alpha, Beta, and Delta
        if fitness<Alpha_score 
            Alpha_score=fitness; % Update alpha
            Alpha_pos=Positions(i,:);
            Alpha_OutputWeight =c(i).OutputWeight;
        end
        
        if fitness>Alpha_score && fitness<Beta_score 
            Beta_score=fitness; % Update beta
            Beta_pos=Positions(i,:);
            Beta_OutputWeight =c(i).OutputWeight;
        end
        
        if fitness>Alpha_score && fitness>Beta_score && fitness<Delta_score 
            Delta_score=fitness; % Update delta
            Delta_pos=Positions(i,:);
            Delta_OutputWeight =c(i).OutputWeight;
        end
    end
    
    
    a=2-l*((2)/Max_iter); % a decreases linearly fron 2 to 0
    
    % Update the Position of search agents including omegas
    for i=1:size(Positions,1)
        for j=1:size(Positions,2)     
                       
            r1=rand(); % r1 is a random number in [0,1]
            r2=rand(); % r2 is a random number in [0,1]
            
            A1=2*a*r1-a; % Equation (3.3)
            C1=2*r2; % Equation (3.4)
            
            D_alpha=abs(C1*Alpha_pos(j)-Positions(i,j)); % Equation (3.5)-part 1
            X1=Alpha_pos(j)-A1*D_alpha; % Equation (3.6)-part 1
                       
            r1=rand();
            r2=rand();
            
            A2=2*a*r1-a; % Equation (3.3)
            C2=2*r2; % Equation (3.4)
            
            D_beta=abs(C2*Beta_pos(j)-Positions(i,j)); % Equation (3.5)-part 2
            X2=Beta_pos(j)-A2*D_beta; % Equation (3.6)-part 2       
            
            r1=rand();
            r2=rand(); 
            
            A3=2*a*r1-a; % Equation (3.3)
            C3=2*r2; % Equation (3.4)
            
            D_delta=abs(C3*Delta_pos(j)-Positions(i,j)); % Equation (3.5)-part 3
            X3=Delta_pos(j)-A3*D_delta; % Equation (3.5)-part 3             
            
            Positions(i,j)=(X1+X2+X3)/3;% Equation (3.7)
            
        end
    end
    l=l+1;
    rmse_training(l,1) = Alpha_score;
    Convergence_curve(l)=Alpha_score;
    [l Alpha_score]
end  % End of main Loop
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Copying results
Best_score=Alpha_score;
Best_pos=Alpha_pos;
GWO_cg_curve=Convergence_curve;
% %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%__________________________________________________________________________ 
%%                       - Testing Results/stage-
%__________________________________________________________________________ 
% In this stage, the parameters und for the perceptron model are tested
% withew/unseen data created by spliting the originaata Such data is
% usually called testing data
%__________________________________________________________________________
% caklculate matrx H for testing data
    
  [ TY ] = model4testing( data4testing, NumberofHiddenNeurons, Alpha_OutputWeight , Alpha_pos, ActivationFunction );
% Computing Root-Mean-Square-Error (RMSE) for testing
  TV.T = label4testing';
  rmse_test=sqrt(mse(TV.T - TY));  
%__________________________________________________________________________
%..........................................................................
%                  - SECTION FOR PLOTTING THE 3D SURFACE -
%..........................................................................
 %%
 varX = 2;
 varY = 3;
 varZ = 4;
 
 res = 50;
% [ X1, Y1, Z1, PC ] = Optimal_region_surfaceB( varX, varY, varZ, X, res, NumberofHiddenNeurons, Alpha_pos, Alpha_OutputWeight, ActivationFunction );
[ X1, Y1, Z1] = Optimal_region_surface( varX, varY, X, res, NumberofHiddenNeurons, Alpha_pos, Alpha_OutputWeight, ActivationFunction );
% [ X1, Z1 ] = Optimal_region_surface_single( varX, X, res, NumberofHiddenNeurons, Alpha_pos, Alpha_OutputWeight, ActivationFunction );
% [ X1, Z1 ] = Optimal_region_surface_singleB( varX, X, res, NumberofHiddenNeurons, Alpha_pos, Alpha_OutputWeight, ActivationFunction );

%--------------------------------------------------------------------------
% Plotting Surface
%%
% tiledlayout(2,1)
% nexttile
for col = 1:res
    for row=1:res
%         X1_new(col) = min_values(1,varX) + ( X1(col)*( max_values(1,varX) - min_values(1,varX) ) );
        X1_new(row,col) = min_values(1,varX) + ( X1(row,col)*( max_values(1,varX) - min_values(1,varX) ) );
        Y1_new(row,col) = min_values(1,varY) + ( Y1(row,col)*( max_values(1,varY) - min_values(1,varY) ) );
    end
end
figure(1)
surf(X1_new,Y1_new,Z1,'FaceAlpha',0.5);

% SingleZ=Z1(:,1);
% SingleX=X1_new(:,1);
% scatter(X1_new,SingleZ);

% figure(1)
% surf(X1,Y1,Z1,'FaceAlpha',0.5); %3D plot for 2 parameters GGIF AND GGIM
% title('Alpha wolf')

xlabel('GGIF') %varX=2
ylabel('GGIM') %varX=3

% xlabel('GGIM') %varX=3
% ylabel('GGIB') %varX=4

% xlabel('GGIB') %varX=4
% ylabel('GGIF') %varX=2



% zlabel('PCE(%)')
figure(2)
plot(TV.T,TV.T)
xlabel('Experiental PCE(%)')
ylabel('Predicted PCE(%)')
hold on
plot(TV.T,TY,'o')
hold off
% surf(X1,Y1,Z1,'FaceAlpha',0.5);

% datX = Paratwo(:,varX);
% datY = Paratwo(:,varY);
% datZ = Paratwo(:,varZ);

% figure(3)
% scatter3 (datX,datY,datZ,40,Paratwo(:,14+Property_cells),'filled')
% ax = gca;
% ax.XDir='reverse';
% view(-31,14)
% cb=colorbar;


% PCMmax = max(max(Z1));
% [val,ida] = max(Z1(:));
% Y1op = Y1(ida);
% X1op = X1(ida);
% [X1op Y1op PCMmax]

% nexttile
% plot(TV.T, TY,'O');

% figure('Position',[500 500 660 290])
% %Draw search space
% subplot(1,2,1);
% func_plot(Function_name);
% title('Parameter space')
% xlabel('x_1');
% ylabel('x_2');
% zlabel([Function_name,'( x_1 , x_2 )'])
% 
% %Draw objective space
% subplot(1,2,2);
% semilogy(GWO_cg_curve,'Color','r')
% title('Objective space')
% xlabel('Iteration');
% ylabel('Best score obtained so far');
% 
% axis tight
% grid on
% box on
% legend('GWO')
% 
% display(['The best solution obtained by GWO is : ', num2str(Best_pos)]);
% display(['The best optimal value of the objective funciton found by GWO is : ', num2str(Best_score)]);
% 
%         
%% Section to calculate the correlation coefficient for training and testing
% Update
% for testing
 R2_testing = rsquare(TV.T',TY');
% 
 R2_training_final = rsquare(T',Y');
 [R2_training_final R2_testing];

 mae_training = mae((T-Y));
 mae_testing = mae((TV.T-TY));
toc

% out = [rmse_training(Max_iter,1),R2_training_final,rmse_test,R2_testing,varX,X1op,varY,Y1op,PCMmax];

% if rmse_test<2
%     out = [rmse_training(Max_iter,1),R2_training_final,rmse_test,R2_testing,varX,X1op,varY,Y1op,PCMmax];
%     writematrix(out,'result of predicted.csv','WriteMode','append');
% end

% such=data4testing(:,2)';
% R3_testing = rsquare(TV.T',data4testing(:,2)');

% Testing the correlations between each variables
% Cor = corrcoef(X_nonN);
% 
% Perform = X_nonN(:,18);
% Pho = X_nonN (:,15:18);
% Cor2 = corr(X_nonN,Pho);
% 
% string_name= {'CGI','GGIF','GGIM','GGIB','AD','PDT','BL','CT','ES','FS','ST','BFL','BFT','CBO','Jsc','Voc','FF','PCE'};
% string_Pname = {'Jsc','Voc','FF','PCE'};
% figure (4)
% xvalu = string_Pname;
% yvalu = string_name;
% h1 = heatmap(xvalu,yvalu,Cor2,'FontSize',12,'FontName','Arial');
