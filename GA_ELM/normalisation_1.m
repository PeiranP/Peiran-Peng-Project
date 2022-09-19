function [ ejnor ] = normalisation_1( M, opcion )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
%--------------------------------------------------------------------------
%                     - PROCESO DE NORMALIZACION -
%--------------------------------------------------------------------------
[filas, columnas] = size(M);
maximo_M = max(M);
minimo_M = min(M);
ejnor = zeros(filas,columnas);
%..........................................................................
if opcion == 1
%==========================================================================
% ejnor1 es para normalizar con opcion 1 (0-1), ejnor2 es para de -1 a 1
for j = 1:columnas  % moverse a lo largo de las dimensiones
    %   Para moverse atraves de las columnas
    for i =1:filas
       if maximo_M(1,j) == minimo_M(1,j)
           % Comparar con 1
           if maximo_M(1,j) >= 1
               ejnor(i,j) = 1;
           % Comparar con 0
           elseif maximo_M(1,j) >= 0
               ejnor(i,j) = -1;
           end    
       else
           ejnor(i,j) = 2*((M(i,j) - minimo_M(1,j))/(maximo_M(1,j) - minimo_M(1,j))) - 1;
       end  % fin de if 
    end
end
%--------------------------------------------------------------------------
else
%==========================================================================
for j = 1:columnas  % moverse a lo largo de las dimensiones
    %   Para moverse atraves de las columnas
    for i =1:filas
       if maximo_M(1,j) == minimo_M(1,j)
           % Comparar con 1
           if maximo_M(1,j) >= 1
               ejnor(i,j) = 1;
           % Comparar con 0
           elseif maximo_M(1,j) >= 0
               ejnor(i,j) = 0;
           end    
       else
           ejnor(i,j) = (M(i,j) - minimo_M(1,j))/(maximo_M(1,j) - minimo_M(1,j));
       end  % fin de if 
    end
end
%--------------------------------------------------------------------------
end

%--------------------------------------------------------------------------
end

