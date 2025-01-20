

load 1d_matlab_data_k_1.mat

main_title = "Solutions for k = 1";

figure(1)

subplot(1, 3, 1)
plot(x_pnts1, exact_u(:, 1), 'r-','DisplayName','Re(\psi(x,t)) Exact','LineWidth',2)
hold on
plot(x_pnts1, nn_pred_u(:, 1), 'b--','DisplayName','Re(\psi(x,t)) MLP','LineWidth',2)

plot(x_pnts1, exact_v(:, 1), 'k-','DisplayName','Im(\psi(x,t)) Exact','LineWidth',2)
plot(x_pnts1, nn_pred_v(:, 1), '--', 'color', [0 0.5 0], 'DisplayName','Im(\psi(x,t)) MLP','LineWidth',2)

temp_title = sprintf('t = %.2f', times(1));
title(temp_title, 'Fontsize', 16)
ylim([-1.1 1.1])
xlim([-2.0 1.0])
ylabel('\psi(x,t)', 'Fontsize', 18)

hold off

subplot(1, 3, 2)
plot(x_pnts2, exact_u(:, 2), 'r-','DisplayName','Re(\psi(x,t)) Exact','LineWidth',2)
hold on
plot(x_pnts2, nn_pred_u(:, 2), 'b--','DisplayName','Re(\psi(x,t)) MLP','LineWidth',2)

plot(x_pnts2, exact_v(:, 2), 'k-','DisplayName','Im(\psi(x,t)) Exact','LineWidth',2)
plot(x_pnts2, nn_pred_v(:, 2), '--', 'color', [0 0.5 0], 'DisplayName','Im(\psi(x,t)) MLP','LineWidth',2)

temp_title = sprintf('t = %.2f', times(2));
title(temp_title, 'Fontsize', 16)
ylim([-1.1 1.1])
xlim([-2.0 1.0])
xlabel('x', 'Fontsize', 18)

hold off

Lgnd = legend('show','Fontsize',13);
Lgnd.Position(1) = -0.06;
Lgnd.Position(2) = 0.712;

subplot(1, 3, 3)
plot(x_pnts3, exact_u(:, 3), 'r-','DisplayName','Re(\psi(x,t)) Exact','LineWidth',2)
hold on
plot(x_pnts3, nn_pred_u(:, 3), 'b--','DisplayName','Re(\psi(x,t)) MLP','LineWidth',2)

plot(x_pnts3, exact_v(:, 3), 'k-','DisplayName','Im(\psi(x,t)) Exact','LineWidth',2)
plot(x_pnts3, nn_pred_v(:, 3), '--', 'color', [0 0.5 0], 'DisplayName','Im(\psi(x,t)) MLP','LineWidth',2)

temp_title = sprintf('t = %.2f', times(3));
title(temp_title, 'Fontsize', 16)
ylim([-1.1 1.1])
xlim([-2.0 1.0])

hold off

sgtitle(main_title, 'Fontsize', 16) 


% figure(2)
% 
% subplot(1, 3, 1)
% plot(x_pnts1, exact_v(:, 1), 'r-','DisplayName','Exact solution','LineWidth',2)
% hold on
% plot(x_pnts1, nn_pred_v(:, 1), 'k--','DisplayName','MLP solution','LineWidth',2)
% temp_title = sprintf('t = %.2f', times(1));
% title(temp_title, 'Fontsize', 16)
% ylim([-1.1 1.1])
% xlim([-2.0 1.0])
% ylabel('Im(\psi(x,t))', 'Fontsize', 18)
% 
% hold off
% 
% subplot(1, 3, 2)
% plot(x_pnts2, exact_v(:, 2), 'r-','DisplayName','Exact solution','LineWidth',2)
% hold on
% plot(x_pnts2, nn_pred_v(:, 2), 'k--','DisplayName','MLP solution','LineWidth',2)
% temp_title = sprintf('t = %.2f', times(2));
% title(temp_title, 'Fontsize', 16)
% ylim([-1.1 1.1])
% xlim([-2.0 1.0])
% xlabel('x', 'Fontsize', 18)
% 
% hold off
% 
% Lgnd = legend('show','Fontsize',13);
% Lgnd.Position(1) = -0.05;
% Lgnd.Position(2) = 0.787;
% 
% subplot(1, 3, 3)
% plot(x_pnts3, exact_v(:, 3), 'r-','DisplayName','Exact solution','LineWidth',2)
% hold on 
% plot(x_pnts3, nn_pred_v(:, 3), 'k--','DisplayName','MLP solution','LineWidth',2)
% temp_title = sprintf('t = %.2f', times(3));
% title(temp_title, 'Fontsize', 16)
% ylim([-1.1 1.1])
% xlim([-2.0 1.0])
% 
% hold off
% sgtitle(main_title, 'Fontsize', 16) 

