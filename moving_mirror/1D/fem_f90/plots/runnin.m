%running the whole FEM method
clear all 
close all
format long e

T = 2;
M = 20480; 
beta = -2;
N= 20480;
acon = .5;

k = pi;                        %wave number 
%alpha = (k^2);

alpha = -1/k;

%solving using a very small time step and acting as if 
%this is the exact solution

eta = .5*ones(64,1);
tic
[uexact,ttil,h] = fem_mover(T,M,N,beta,acon,alpha,k,eta);
toc
h_vec = [20480,20,40,80,160,320,640,1280,2560,5120,10240]; 
max_err = zeros(11,1);
Err_msq = zeros(11,1); 
count = 0;
for i=2:length(h_vec)
    N = h_vec(i);
    tic
    [U,ttil,h] = fem_mover(T,M,N,beta,acon,alpha,k,eta);
    toc;
    %choosing the grid comparison
    comp1 = (h_vec(1))/(h_vec(i));
    
    %finding the error of the problem
    temp = uexact(1:comp1:end,:);

    Err = abs(U - temp);  

    %finding the maximum nodal error 
     max_err(i-1) = max(max(Err));
    
    %computing the mean-squared nodal error 
    temp2 = 0; 
    for ktil = 1:length(ttil)
        temp = 0;
        for mtil = 1:N+1
            temp = (Err(mtil,ktil))^2 + temp; 
        end
        temp2 = (1/(N+1))*sqrt(temp) + temp2;
    end
    
    Err_msq(i-1) = (1/length(ttil))*temp2;
    count = count + 1;
    whos

    clearvars -except T M beta N acon k alpha eta uexact h_vec max_err ...
        Err_msq count i
end


max_err

Err_msq

%finding the estimated rate of convergence for the max nodal error 
EOC_max(1,1) = 0;
for i = 2:count
EOC_max(i,1) = log(max_err(i-1,1)/max_err(i,1))/log(2); 
end

EOC_max

%finding the estimated rate of convergence for the mean squared nodal error 
EOC_msq(1,1) = 0;
for i = 2:count
EOC_msq(i,1) = log(Err_msq(i-1,1)/Err_msq(i,1))/log(2); 
end

EOC_msq


