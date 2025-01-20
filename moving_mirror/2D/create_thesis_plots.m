close all 
clear all

load 2d_matlab_data_k_3.mat

main_title = "Re(\psi(x,y,t)) MLP Solutions for k = 3";

figure(1)

subplot(1, 3, 1)
surf(x_pnts1, y_pnts1, nn_pred_u(:, :, 1),'DisplayName','Re(\psi(x,y,t)) MLP','LineWidth',2)
shading interp;
temp_title = sprintf('t = %.2f', times(1));
title(temp_title, 'Fontsize', 16)
ylim([-2.0 1.0])
xlim([-2.0 1.0])
zlim([-1.5 1.5])
zlabel('\psi(x,y,t)', 'Fontsize', 18)
xlabel('x', 'Fontsize', 18)
ylabel('y', 'Fontsize', 18)
view([19 41])

subplot(1, 3, 2)
surf(x_pnts2, y_pnts2, nn_pred_u(:, :, 2),'DisplayName','Re(\psi(x,y,t)) MLP','LineWidth',2)
shading interp;
temp_title = sprintf('t = %.2f', times(2));
title(temp_title, 'Fontsize', 16)
ylim([-2.0 1.0])
xlim([-2.0 1.0])
zlim([-1.5 1.5])
zlabel('\psi(x,y,t)', 'Fontsize', 18)
xlabel('x', 'Fontsize', 18)
ylabel('y', 'Fontsize', 18)
view([19 41])

subplot(1, 3, 3)
surf(x_pnts3, y_pnts3, nn_pred_u(:, :, 3),'DisplayName','Re(\psi(x,y,t)) MLP','LineWidth',2)
shading interp;
temp_title = sprintf('t = %.2f', times(3));
title(temp_title, 'Fontsize', 16)
ylim([-2.0 1.0])
xlim([-2.0 1.0])
zlim([-1.5 1.5])
zlabel('\psi(x,y,t)', 'Fontsize', 18)
xlabel('x', 'Fontsize', 18)
ylabel('y', 'Fontsize', 18)
view([19 41])


sgtitle(main_title, 'Fontsize', 16) 


main_title = "Im(\psi(x,y,t)) MLP Solutions for k = 3";

figure(2)

subplot(1, 3, 1)
surf(x_pnts1, y_pnts1, nn_pred_v(:, :, 1),'DisplayName','Im(\psi(x,y,t)) MLP','LineWidth',2)
shading interp;
temp_title = sprintf('t = %.2f', times(1));
title(temp_title, 'Fontsize', 16)
ylim([-2.0 1.0])
xlim([-2.0 1.0])
zlim([-1.5 1.5])
zlabel('\psi(x,y,t)', 'Fontsize', 18)
xlabel('x', 'Fontsize', 18)
ylabel('y', 'Fontsize', 18)
view([19 41])

subplot(1, 3, 2)
surf(x_pnts2, y_pnts2, nn_pred_v(:, :, 2),'DisplayName','Im(\psi(x,y,t)) MLP','LineWidth',2)
shading interp;
temp_title = sprintf('t = %.2f', times(2));
title(temp_title, 'Fontsize', 16)
ylim([-2.0 1.0])
xlim([-2.0 1.0])
zlim([-1.5 1.5])
zlabel('\psi(x,y,t)', 'Fontsize', 18)
xlabel('x', 'Fontsize', 18)
ylabel('y', 'Fontsize', 18)
view([19 41])

subplot(1, 3, 3)
surf(x_pnts3, y_pnts3, nn_pred_v(:, :, 3),'DisplayName','Im(\psi(x,y,t)) MLP','LineWidth',2)
shading interp;
temp_title = sprintf('t = %.2f', times(3));
title(temp_title, 'Fontsize', 16)
ylim([-2.0 1.0])
xlim([-2.0 1.0])
zlim([-1.5 1.5])
zlabel('\psi(x,y,t)', 'Fontsize', 18)
xlabel('x', 'Fontsize', 18)
ylabel('y', 'Fontsize', 18)
view([19 41])


sgtitle(main_title, 'Fontsize', 16) 
