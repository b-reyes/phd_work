%moving FEM 
function [U,t,hm2] = fem_mover(T,M,N,beta,acon,alpha,k,eta)

%creating t from M and T
dt = T/M;
t = dt*[0:M];

%gamma function that determines the right boundary 
gamma = @(t) acon*t^2;

%allocating a vector for all the spacial steps h
hm1 = zeros(1,M+1);
hm = zeros(1,M+1);
hm2 = zeros(1,M+1);
%U matrix full of our unknown constants
U = zeros(N+1,M+1);

G = zeros(N-1,1);

G(1,1) = 1;

G(N-1,1) = 1;

n = 1:N+1;

%creating our step size for time 1
h = (gamma(t(1)) - beta)/N;
    
%creating the time dependent spacial domain for 1
a = beta + (n - 1)*h;

summer = zeros(length(a),1)';

%finding the sum
for i=1:length(eta)
	summer = summer + (eta(i,1)/i^2)*sin(i*pi*(a - beta)/(-beta));
end

%initializing the first entry of U
U(:,1) = (exp(1i*k*a)).*(a - beta).*(a - gamma(t(1))) + summer;

for m = 2:M+1
    
    %creating our step size for the m-1 time 
    hm1(m-1) = (gamma((m-1)*dt) - beta)/(N);
    
    %creating the time dependent spacial domain for m-1
    am1 = beta + (n - 1)*hm1(m-1);
    
    %creating our step size for the mth time 
    hm2(m-1) = (gamma((m-.5)*dt) - beta)/(N);
    
    %creating the time dependent spacial domain for m time 
    a_vec = beta + (n - 1)*hm2(m-1);
         
    hm(m-1) = (gamma((m)*dt) - beta)/(N);                             
    am = beta + (n - 1)*hm(m-1);
    
    %creating the matrices needed for each time
    A = a_matrix(a_vec);
    M = m_matrix(a_vec);
    Q = q_matrix(a_vec,am,am1);
    
    %creating the left hand side matrix of the equation
    %LHS = 1i*M(2:N,2:N) -(1i/2)*Q(2:N,2:N) -((alpha*dt)/2)*A(2:N,2:N); 
    LHS = 1i*M -(1i/2)*Q -((alpha*dt)/2)*A; 
    
    G(1,1) = -(1i*(a_vec(2)-a_vec(2-1))/6 - (1i/2)*((2*am(1) - 2*am1(1) + am(2) - am1(2))./6) -((alpha*dt)/2)*(-1./(a_vec(2)-a_vec(2-1))))*U(1,m) + ....
        (1i*(a_vec(2)-a_vec(2-1))/6 + (1i/2)*((2*am(1) - 2*am1(1) + am(2) - am1(2))./6) + ((alpha*dt)/2)*(-1./(a_vec(2)-a_vec(2-1))))*U(1,m-1);
    
    G(end,1) = -(1i*((a_vec(N+1)-a_vec(N))/6)-(1i/2)*((-2*am(N+1) + 2*am1(N+1) - am(N) + am1(N))/6)-((alpha*dt)/2)*(-1./(a_vec(N+1)-a_vec(N))))*U(N+1,m) + ...
        (1i*((a_vec(N+1)-a_vec(N))/6)+(1i/2)*((-2*am(N+1) + 2*am1(N+1) - am(N) + am1(N))/6)+((alpha*dt)/2)*(-1./(a_vec(N+1)-a_vec(N))))*U(N+1,m-1);
    
    %creating the right hand side of the equation
    %RHS = (1i*M(2:N,2:N) + (1i/2)*Q(2:N,2:N) + ((alpha*dt)/2)*A(2:N,2:N))*U(2:N,m-1) + G; 
    RHS = (1i*M + (1i/2)*Q + ((alpha*dt)/2)*A)*U(2:N,m-1) + G; 
    %solving for the next time step 
    U(2:N,m) = LHS\RHS;
end
