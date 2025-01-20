%Graphing the FEM method
clear all 
close all

T = 2;
M = 10;
beta = -2;
N= 10;
start_a = -.5;
end_a = .5; 

k = pi;                        %wave number 
alpha = k^2;

%solving using a very small time step and acting as if 
%this is the exact solution
acon = (end_a-start_a).*rand(1,1) + start_a;
acon = acon + 1.5;
eta = (end_a-start_a).*rand(64,1) + start_a;

[uexact,ttil,h] = fem_mover(T,M,N,beta,acon,alpha,k,eta);

uexact = real(uexact).^2;

n = 1:N+1;

subplot(1,2,1)
for k=1:length(ttil)
    a_vec = beta + (n - 1)*h(k);
    hold on
    plot(a_vec,ttil(k)*ones(length(a_vec),1),'k.','MarkerSize',20)
    hold off
    xlim([-2,8])
    ylim([-0.1,1.95])
    xlabel('Spatial Nodal Points (x^m)')
    ylabel('t_m')
end 

%solving using a very small time step and acting as if 
%this is the exact solution
acon = (end_a-start_a).*rand(1,1) + start_a;
acon = acon + 1.5;
eta = (end_a-start_a).*rand(64,1) + start_a;

[uexact,ttil,h] = fem_mover(T,M,N,beta,acon,alpha,k,eta);

subplot(1,2,2)

for k=1:length(ttil)
    a_vec = beta + (n - 1)*h(k);
    hold on
    plot(a_vec,ttil(k)*ones(length(a_vec),1),'k.','MarkerSize',20)
    hold off
    xlim([-2,8])
    ylim([-0.1,1.95])
    xlabel('Spatial Nodal Points (x^m)')
    ylabel('t_m')
end 





