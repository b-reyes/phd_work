%Graphing the FEM method
clear all 
close all  

T = 2;
M = 160;
beta = -2;
N= 160;
start_a = -.5;
end_a = .5; 

k = pi;                        %wave number 
alpha = -1/k;

%solving using a very small time step and acting as if 
%this is the exact solution
acon = (end_a-start_a).*rand(1,1) + start_a;
acon = acon + 1.5;
eta = (end_a-start_a).*rand(64,1) + start_a;

[uexact,ttil,h] = fem_mover(T,M,N,beta,acon,alpha,k,eta);

uexact = abs(uexact).^2;

n = 1:N+1;
a_vec = beta + (n - 1)*h(1);    
plot(a_vec,uexact(:,1),'k-')
hold on
a_vec = beta + (n - 1)*h(floor(length(ttil)/2));    
plot(a_vec,uexact(:,floor(length(ttil)/2)),'k--')

a_vec = beta + (n - 1)*h(length(ttil)-1);    
plot(a_vec,uexact(:,length(ttil)-1),'k-.')

hold off
 
legend('t = 0','t = 0.9875','t = 1.9875')
xlabel('Time-dependent spatial domain')
ylabel('| \psi(x,t) |^2')
title('Density Profile')






