%creating the A matrix 
function [A] = a_matrix(x) 

n = length(x);

j = 2:n-1;

A1 = ones(n-2,1);
A2 = ones(n-2,1);
A3 = ones(n-2,1);

%main diagonal
A1(1:n-2,1) =  1./(x(j) - x(j-1)) + 1./(x(j+1) - x(j));

%super diagonal
A2(2:n-1,1) = -1./(x(j+1)-x(j));

%sub diagonal
A3(1:n-2,1) = -1./(x(j)-x(j-1));  

% A1(1,1) = 1/(x(2)-x(1));
% 
% A2(2,1) = (-1)/(x(2)-x(1));
% 
% A1(n,1) = 1/(x(n)-x(n-1));
% 
% A3(n-1,1) = (-1)/(x(n)-x(n-1));
 
A = spdiags([A3(1:n-2,1),A1(1:n-2,1),A2(1:n-2,1)],[-1,0,1],n-2,n-2);
                                                                                                                                       
end 
