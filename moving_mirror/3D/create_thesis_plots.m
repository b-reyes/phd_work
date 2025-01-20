close all 
clear all

% load /Users/brandonreyes/Desktop/3d_matlab_data_k_1.mat
load 3d_matlab_data_k_1.mat
acon_z = 0.25;

main_title = "Re(\psi(x,y,z,t)) MLP Solutions for k = 1";

figure(1)

subplot(1, 3, 1)
[X, Y, Z] = meshgrid(x_pnts1, y_pnts1, z_pnts1);
[xslice, yslice] = meshgrid(x_pnts1, y_pnts1);
zslice = xslice; %(xslice + yslice)/2;
slice(X, Y, Z, nn_pred_u(:, :, :, 1), xslice, yslice, zslice)    % display the slices
shading interp;
hold on
slice(X, Y, Z, nn_pred_u(:, :, :, 1),[],[],(-2.0 + acon_z*times(1)^2)/2)
shading interp;
hold off
temp_title = sprintf('t = %.2f', times(1));
title(temp_title, 'Fontsize', 16)
xlim([-2 1.1])
ylim([-2 1.1])
zlim([-2 1.1])
xlabel('x', 'Fontsize', 18)
ylabel('y', 'Fontsize', 18)
zlabel('z', 'Fontsize', 18)
view(-34,24)


subplot(1, 3, 2)
[X, Y, Z] = meshgrid(x_pnts2, y_pnts2, z_pnts2);
[xslice, yslice] = meshgrid(x_pnts2, y_pnts2);
zslice = xslice; %(xslice + yslice)/2;
slice(X, Y, Z, nn_pred_u(:, :, :, 2), xslice, yslice, zslice)    % display the slices
shading interp;
hold on
slice(X, Y, Z, nn_pred_u(:, :, :, 2),[],[],(-2.0 + acon_z*times(2)^2)/2)
shading interp;
hold off
temp_title = sprintf('t = %.2f', times(2));
title(temp_title, 'Fontsize', 16)
xlim([-2 1.1])
ylim([-2 1.1])
zlim([-2 1.1])
xlabel('x', 'Fontsize', 18)
ylabel('y', 'Fontsize', 18)
zlabel('z', 'Fontsize', 18)
view(-34,24)

subplot(1, 3, 3)
[X, Y, Z] = meshgrid(x_pnts3, y_pnts3, z_pnts3);
[xslice, yslice] = meshgrid(x_pnts3, y_pnts3);
zslice = xslice; %(xslice + yslice)/2;
slice(X, Y, Z, nn_pred_u(:, :, :, 3), xslice, yslice, zslice)    % display the slices
shading interp;
hold on
slice(X, Y, Z, nn_pred_u(:, :, :, 3),[],[],(-2.0 + acon_z*times(3)^2)/2)
shading interp;
hold off
temp_title = sprintf('t = %.2f', times(3));
title(temp_title, 'Fontsize', 16)
xlim([-2 1.1])
ylim([-2 1.1])
zlim([-2 1.1])
xlabel('x', 'Fontsize', 18)
ylabel('y', 'Fontsize', 18)
zlabel('z', 'Fontsize', 18)
view(-34,24)
h=colorbar;
set(h, 'Position', [0.94736 0.150346 0.0152 0.70615])


sgtitle(main_title, 'Fontsize', 16) 


main_title = "Im(\psi(x,y,z,t)) MLP Solutions for k = 1";

figure(2)

subplot(1, 3, 1)
[X, Y, Z] = meshgrid(x_pnts1, y_pnts1, z_pnts1);
[xslice, yslice] = meshgrid(x_pnts1, y_pnts1);
zslice = xslice; %(xslice + yslice)/2;
slice(X, Y, Z, nn_pred_v(:, :, :, 1), xslice, yslice, zslice)    % display the slices
shading interp;
hold on
slice(X, Y, Z, nn_pred_v(:, :, :, 1),[],[],(-2.0 + acon_z*times(1)^2)/2)
shading interp;
hold off
temp_title = sprintf('t = %.2f', times(1));
title(temp_title, 'Fontsize', 16)
xlim([-2 1.1])
ylim([-2 1.1])
zlim([-2 1.1])
xlabel('x', 'Fontsize', 18)
ylabel('y', 'Fontsize', 18)
zlabel('z', 'Fontsize', 18)
view(-34,24)


subplot(1, 3, 2)
[X, Y, Z] = meshgrid(x_pnts2, y_pnts2, z_pnts2);
[xslice, yslice] = meshgrid(x_pnts2, y_pnts2);
zslice = xslice; %(xslice + yslice)/2;
slice(X, Y, Z, nn_pred_v(:, :, :, 2), xslice, yslice, zslice)    % display the slices
shading interp;
hold on
slice(X, Y, Z, nn_pred_v(:, :, :, 2),[],[],(-2.0 + acon_z*times(2)^2)/2)
shading interp;
hold off
temp_title = sprintf('t = %.2f', times(2));
title(temp_title, 'Fontsize', 16)
xlim([-2 1.1])
ylim([-2 1.1])
zlim([-2 1.1])
xlabel('x', 'Fontsize', 18)
ylabel('y', 'Fontsize', 18)
zlabel('z', 'Fontsize', 18)
view(-34,24)

subplot(1, 3, 3)
[X, Y, Z] = meshgrid(x_pnts3, y_pnts3, z_pnts3);
[xslice, yslice] = meshgrid(x_pnts3, y_pnts3);
zslice = xslice; %(xslice + yslice)/2;
slice(X, Y, Z, nn_pred_v(:, :, :, 3), xslice, yslice, zslice)    % display the slices
shading interp;
hold on
slice(X, Y, Z, nn_pred_v(:, :, :, 3),[],[],(-2.0 + acon_z*times(3)^2)/2)
shading interp;
hold off
temp_title = sprintf('t = %.2f', times(3));
title(temp_title, 'Fontsize', 16)
xlim([-2 1.1])
ylim([-2 1.1])
zlim([-2 1.1])
xlabel('x', 'Fontsize', 18)
ylabel('y', 'Fontsize', 18)
zlabel('z', 'Fontsize', 18)
view(-34,24)
h=colorbar;
set(h, 'Position', [0.94736 0.150346 0.0152 0.70615])

sgtitle(main_title, 'Fontsize', 16) 

k = 3;
w = 1.0; 
k_c = 1.0;
beta = -2.0;
acon_x = 0.25;
acon_y = 0.25;
acon_z = 0.25;

main_title = "Re(\psi(x,y,z,t)) Exact Solutions for k = 3";

figure(3)

subplot(1, 3, 1)
[X, Y, Z] = meshgrid(x_pnts1, y_pnts1, z_pnts1);
[xslice, yslice] = meshgrid(x_pnts1, y_pnts1);
zslice = xslice; %(xslice + yslice)/2;
[u_exact_real1, u_exact_imag1] = exact_soln(X, Y, Z, k, w, k_c, beta, acon_x, acon_y, acon_z, times(1));
slice(X, Y, Z, u_exact_real1, xslice, yslice, zslice)    % display the slices
shading interp;
hold on
slice(X, Y, Z, u_exact_real1,[],[],(-2.0 + acon_z*times(1)^2)/2)
shading interp;
hold off
temp_title = sprintf('t = %.2f', times(1));
title(temp_title, 'Fontsize', 16)
xlim([-2 1.1])
ylim([-2 1.1])
zlim([-2 1.1])
xlabel('x', 'Fontsize', 18)
ylabel('y', 'Fontsize', 18)
zlabel('z', 'Fontsize', 18)
view(-34,24)


subplot(1, 3, 2)
[X, Y, Z] = meshgrid(x_pnts2, y_pnts2, z_pnts2);
[xslice, yslice] = meshgrid(x_pnts2, y_pnts2);
zslice = xslice; %(xslice + yslice)/2;
[u_exact_real2, u_exact_imag2] = exact_soln(X, Y, Z, k, w, k_c, beta, acon_x, acon_y, acon_z, times(2));
slice(X, Y, Z, u_exact_real2, xslice, yslice, zslice)    % display the slices
shading interp;
hold on
slice(X, Y, Z, u_exact_real2,[],[],(-2.0 + acon_z*times(2)^2)/2)
shading interp;
hold off
temp_title = sprintf('t = %.2f', times(2));
title(temp_title, 'Fontsize', 16)
xlim([-2 1.1])
ylim([-2 1.1])
zlim([-2 1.1])
xlabel('x', 'Fontsize', 18)
ylabel('y', 'Fontsize', 18)
zlabel('z', 'Fontsize', 18)
view(-34,24)

subplot(1, 3, 3)
[X, Y, Z] = meshgrid(x_pnts3, y_pnts3, z_pnts3);
[xslice, yslice] = meshgrid(x_pnts3, y_pnts3);
zslice = xslice; %(xslice + yslice)/2;
[u_exact_real3, u_exact_imag3] = exact_soln(X, Y, Z, k, w, k_c, beta, acon_x, acon_y, acon_z, times(3));
slice(X, Y, Z, u_exact_real3, xslice, yslice, zslice)    % display the slices
shading interp;
hold on
slice(X, Y, Z, u_exact_real3,[],[],(-2.0 + acon_z*times(3)^2)/2)
shading interp;
hold off
temp_title = sprintf('t = %.2f', times(3));
title(temp_title, 'Fontsize', 16)
xlim([-2 1.1])
ylim([-2 1.1])
zlim([-2 1.1])
xlabel('x', 'Fontsize', 18)
ylabel('y', 'Fontsize', 18)
zlabel('z', 'Fontsize', 18)
view(-34,24)
h=colorbar;
set(h, 'Position', [0.94736 0.150346 0.0152 0.70615])


sgtitle(main_title, 'Fontsize', 16) 


main_title = "Im(\psi(x,y,z,t)) Exact Solutions for k = 3";

figure(4)

subplot(1, 3, 1)
[X, Y, Z] = meshgrid(x_pnts1, y_pnts1, z_pnts1);
[xslice, yslice] = meshgrid(x_pnts1, y_pnts1);
zslice = xslice; %(xslice + yslice)/2;
slice(X, Y, Z, u_exact_imag1, xslice, yslice, zslice)    % display the slices
shading interp;
hold on
slice(X, Y, Z, u_exact_imag1,[],[],(-2.0 + acon_z*times(1)^2)/2)
shading interp;
hold off
temp_title = sprintf('t = %.2f', times(1));
title(temp_title, 'Fontsize', 16)
xlim([-2 1.1])
ylim([-2 1.1])
zlim([-2 1.1])
xlabel('x', 'Fontsize', 18)
ylabel('y', 'Fontsize', 18)
zlabel('z', 'Fontsize', 18)
view(-34,24)


subplot(1, 3, 2)
[X, Y, Z] = meshgrid(x_pnts2, y_pnts2, z_pnts2);
[xslice, yslice] = meshgrid(x_pnts2, y_pnts2);
zslice = xslice; %(xslice + yslice)/2;
slice(X, Y, Z, u_exact_imag2, xslice, yslice, zslice)    % display the slices
shading interp;
hold on
slice(X, Y, Z, u_exact_imag2,[],[],(-2.0 + acon_z*times(2)^2)/2)
shading interp;
hold off
temp_title = sprintf('t = %.2f', times(2));
title(temp_title, 'Fontsize', 16)
xlim([-2 1.1])
ylim([-2 1.1])
zlim([-2 1.1])
xlabel('x', 'Fontsize', 18)
ylabel('y', 'Fontsize', 18)
zlabel('z', 'Fontsize', 18)
view(-34,24)

subplot(1, 3, 3)
[X, Y, Z] = meshgrid(x_pnts3, y_pnts3, z_pnts3);
[xslice, yslice] = meshgrid(x_pnts3, y_pnts3);
zslice = xslice; %(xslice + yslice)/2;
slice(X, Y, Z, u_exact_imag3, xslice, yslice, zslice)    % display the slices
shading interp;
hold on
slice(X, Y, Z, u_exact_imag3,[],[],(-2.0 + acon_z*times(3)^2)/2)
shading interp;
hold off
temp_title = sprintf('t = %.2f', times(3));
title(temp_title, 'Fontsize', 16)
xlim([-2 1.1])
ylim([-2 1.1])
zlim([-2 1.1])
xlabel('x', 'Fontsize', 18)
ylabel('y', 'Fontsize', 18)
zlabel('z', 'Fontsize', 18)
view(-34,24)
h=colorbar;
set(h, 'Position', [0.94736 0.150346 0.0152 0.70615])

sgtitle(main_title, 'Fontsize', 16) 


function [u_exact_real, u_exact_imag] = exact_soln(XX, YY, ZZ, k, w, k_c, beta, acon_x, acon_y, acon_z, t)

    val = exp(1i*(k*XX + k*YY + k*ZZ - w*t)).*sin(k_c*pi*(XX - beta).*...
            (XX-acon_x*t^2).*(YY - beta).*(YY-acon_y*t^2).*(ZZ - beta).*(ZZ-acon_z*t^2));

    u_exact_real = real(val);
    u_exact_imag = imag(val);

end


