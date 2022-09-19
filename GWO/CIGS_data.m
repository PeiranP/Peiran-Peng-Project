tic
close all
clear
clc

Paraall = readtable('CIGS solar cells.xlsx');
Paraall = rmmissing(Paraall);
ParaPer = readtable('CIGS solar cells.xlsx', 'Range','B3:S382','ReadVariableNames',false);
ParaPer = rmmissing(ParaPer);
% ParaPer = fillmissing (ParaPer,'constant',0,'DataVariables',@isnumeric); %replace Nan with 0


L = height(ParaPer); %Select data with performance higher or equal to 10%
i=0;
PerfR = ParaPer.Var18 >=10;
ParaPer= ParaPer(PerfR,:);

Paraone = ParaPer(:,1:14);
Perf = ParaPer(:,15:18);
% Jud = {'Y','N'};
% JudN = {'1','0'};
Paraone.Var5(strcmpi(Paraone.Var5,'Y')) = {1}; 
Paraone.Var5(strcmpi(Paraone.Var5,'N')) = {0}; 
Paraone.Var6(strcmpi(Paraone.Var6,'Y')) = {1}; 
Paraone.Var6(strcmpi(Paraone.Var6,'N')) = {0}; 
Paraone.Var7(strcmpi(Paraone.Var7,'Y')) = {1}; 
Paraone.Var7(strcmpi(Paraone.Var7,'N')) = {0}; 
Paraone.Var9 = findgroups(Paraone.Var9);
Paraone.Var10 = findgroups(Paraone.Var10);
Paraone.Var12 = findgroups(Paraone.Var12);



W=width(Paraone);
i= 0;
while i<width(Paraone) %could be a function
    i=i+1;
    if iscell(Paraone.(i))
        Paraone.(i)=cell2mat(Paraone.(i));  
    end
end
Paratwo=table2array(Paraone);
% Paratwo=normalize(Paratwo,'range');
Perf = table2array(Perf);

Paratwo=[Paratwo,Perf];%combine parameter and performance
Paratwo = Paratwo(randperm(size(Paratwo,1)),:);% swap matrix row randomly
save CIGS_solarC.mat Paratwo
% Xload = normalize (Xload,'range');
% X = [Xload, species]; %combine character and species together
% X = X(randperm(size(X,1)),:); % swap matrix row randomly