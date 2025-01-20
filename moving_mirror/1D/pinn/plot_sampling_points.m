close all 
clear all 

load 1d_collocation_points.mat

main_title = "Collocation Points in 1D";

figure(1)

plot(t_pnts, x_pnts, 'b.','MarkerSize',10)
hold on
plot(t_pnts, -2.5, 'r.','MarkerSize',14)
hold off
title(main_title, 'Fontsize', 16)
xlim([0 1.6])
ylim([-2.5 1.0])
xlabel('t', 'Fontsize', 18)
ylabel('x', 'Fontsize', 18)
