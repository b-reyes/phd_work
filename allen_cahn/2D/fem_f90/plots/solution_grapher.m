clear all
close all 

fileID = fopen('u_0.txt','r');
formatSpec = '%f';
sizeA = [1281 1281];
U(:,:,1) = fscanf(fileID,formatSpec,sizeA);

figure(1) 
N_ex = 1281;
x_start = 0;
x_end = .5;
hf = (x_end-x_start)/(N_ex-1);
omega_x = hf*[0:N_ex-1];
omega_y = omega_x; 

subplot(1,5,1)
contour(omega_x,omega_y,U(:,:,1),[0 0],'LineColor','b')


fileID = fopen('u_1_4.txt','r');
formatSpec = '%f';
U(:,:,2) = fscanf(fileID,formatSpec,sizeA);
subplot(1,5,2)
contour(omega_x,omega_y,U(:,:,2),[0 0],'LineColor','b')

fileID = fopen('u_1_2.txt','r');
formatSpec = '%f';
U(:,:,3) = fscanf(fileID,formatSpec,sizeA);
subplot(1,5,3)
contour(omega_x,omega_y,U(:,:,3),[0 0],'LineColor','b')

fileID = fopen('u_3_4.txt','r');
formatSpec = '%f';
U(:,:,4) = fscanf(fileID,formatSpec,sizeA);
subplot(1,5,4)
contour(omega_x,omega_y,U(:,:,4),[0 0],'LineColor','b')

fileID = fopen('u_end.txt','r');
formatSpec = '%f';
U(:,:,5) = fscanf(fileID,formatSpec,sizeA);
subplot(1,5,5)
contour(omega_x,omega_y,U(:,:,5),[0 0],'LineColor','b')


