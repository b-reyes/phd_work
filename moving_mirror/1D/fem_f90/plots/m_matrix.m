%creating the M matrix
function M = m_matrix(x)

n = length(x);

j = 2:n-1;

M1 = ones(n-2,1);
M2 = ones(n-2,1);
M3 = ones(n-2,1);

%main diagonal
M1(1:n-2,1) = (x(j+1) - x(j-1))./3;
%super diagonal
M2(2:n-1,1) = (x(j+1)-x(j))./6;
%sub diagonal
M3(1:n-2,1) = (x(j)-x(j-1))./6;

% M1(1,1) = (x(2)-x(1))/3;
% 
% M2(2,1) = (x(2)-x(1))/6;
% 
% M1(n,1) = (x(n)-x(n-1))/3;
% 
% M3(n-1,1) = (x(n)-x(n-1))/6;
 
M = spdiags([M3(1:n-2,1),M1(1:n-2,1),M2(1:n-2,1)],[-1,0,1],n-2,n-2);
                     
end    
