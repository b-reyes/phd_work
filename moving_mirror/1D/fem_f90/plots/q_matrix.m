%creating the Q matrix
function Q = q_matrix(x,xm,xm1)

n = length(x);

j = 2:n-1;

Q1 = ones(n-2,1);
Q2 = ones(n-2,1);
Q3 = ones(n-2,1);

%main diagonal
Q1(1:n-2,1) = (xm(j-1) - xm1(j-1) - xm(j+1) + xm1(j+1))./6;
%super diagonal
Q2(2:n-1,1) = (2*xm(j) - 2*xm1(j) + xm(j+1) - xm1(j+1))./6;
%sub diagonal
Q3(1:n-2,1) = (-2*xm(j) + 2*xm1(j)- xm(j-1) + xm1(j-1))./6;

% Q1(1,1) = (2*xm1(1) - 2*xm(1) - xm(2) + xm1(2))/6;
% 
% Q2(2,1) = (2*xm1(2) - 2*xm(2) - xm(1) + xm1(1))/6;
% 
% Q1(n,1) = (2*xm(n) - 2*xm1(n) + xm(n-1) - xm1(n-1))/6;
% 
% Q3(n-1,1) = (2*xm(n-1) - 2*xm1(n-1) + xm(n) - xm1(n))/6;


Q = spdiags([Q3(1:n-2,1),Q1(1:n-2,1),Q2(1:n-2,1)],[-1,0,1],n-2,n-2);
