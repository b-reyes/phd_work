clear all
close all
dim = 81; 

fileID = fopen('u_0.txt','r');
formatSpec = '%f';
sizeA = [dim dim];
for i= 1:dim
    U(:,:,i,1) = fscanf(fileID,formatSpec,sizeA);
end 

fileID = fopen('u_1_4.txt','r');
formatSpec = '%f';
sizeA = [dim dim];
for i= 1:dim
U(:,:,i,2) = fscanf(fileID,formatSpec,sizeA);
end 

fileID = fopen('u_1_2.txt','r');
formatSpec = '%f';
sizeA = [dim dim];
for i= 1:dim
U(:,:,i,3) = fscanf(fileID,formatSpec,sizeA);
end 

fileID = fopen('u_3_4.txt','r');
formatSpec = '%f';
sizeA = [dim dim];
for i= 1:dim
U(:,:,i,4) = fscanf(fileID,formatSpec,sizeA);
end 

fileID = fopen('u_end.txt','r');
formatSpec = '%f';
sizeA = [dim dim];
for i= 1:dim
U(:,:,i,5) = fscanf(fileID,formatSpec,sizeA);
end 

N_ex = dim;
x_start = 0;
x_end = .5;
hf = (x_end-x_start)/(N_ex-1);
omega_x = hf*[0:N_ex-1];
omega_y = omega_x; 
omega_z = omega_x;

fig_1 = figure;

subplot(1,5,1)
p = patch(isosurface(omega_x,omega_y,omega_z,(U(:,:,:,1)),0));
isonormals(omega_x,omega_y,omega_z,(U(:,:,:,1)),p)
p.FaceColor = 'red';
p.EdgeColor = 'none';
daspect([1,1,1])
view(3); axis tight
camlight 
lighting gouraud

subplot(1,5,2)
p = patch(isosurface(omega_x,omega_y,omega_z,(U(:,:,:,2)),0));
isonormals(omega_x,omega_y,omega_z,(U(:,:,:,2)),p)
p.FaceColor = 'red';
p.EdgeColor = 'none';
daspect([1,1,1])
view(3); axis tight
camlight 
lighting gouraud

subplot(1,5,3)
p = patch(isosurface(omega_x,omega_y,omega_z,(U(:,:,:,3)),0));
isonormals(omega_x,omega_y,omega_z,(U(:,:,:,3)),p)
p.FaceColor = 'red';
p.EdgeColor = 'none';
daspect([1,1,1])
view(3); axis tight
camlight 
lighting gouraud

subplot(1,5,4)
p = patch(isosurface(omega_x,omega_y,omega_z,(U(:,:,:,4)),0));
isonormals(omega_x,omega_y,omega_z,(U(:,:,:,4)),p)
p.FaceColor = 'red';
p.EdgeColor = 'none';
daspect([1,1,1])
view(3); axis tight
camlight 
lighting gouraud


subplot(1,5,5)
p = patch(isosurface(omega_x,omega_y,omega_z,(U(:,:,:,5)),0));
isonormals(omega_x,omega_y,omega_z,(U(:,:,:,5)),p)
p.FaceColor = 'red';
p.EdgeColor = 'none';
daspect([1,1,1])
view(3); axis tight
camlight 
lighting gouraud
