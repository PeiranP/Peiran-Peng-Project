clear;
clc
load f_el.mat
load f_ys.mat
load predicted_value.mat
load error_el.mat


[f_ys1,ind]=sort(f_ys);
[f_el1,ind]=sort(f_el);
TY = zeros(size(f_el,1),1);
%     for row=1:size(f_el,1)
%         TY(row,1) = predicted_value(f_el(row,1),1);
%     end
