close all 
clear all

load 2d_matlab_movie_data_k_3.mat

vidObj = VideoWriter('2D_moving_mirror_movie.mp4', 'MPEG-4');
vidObj.FrameRate = 15;
open(vidObj);
fh = figure;

N_ex = 161;

for k=1:N_ex

    main_title = "Re(\psi(x,y,t)) MLP Solutions for k = 3";

    subplot(1, 2, 1)
    surf(all_xpnts(:, :, k), all_ypnts(:, :, k), nn_pred_u(:, :, k),'DisplayName','Re(\psi(x,y,t)) MLP','LineWidth',2)
    shading interp;
    temp_title = sprintf('t = %.2f', times(k));
    title(temp_title, 'Fontsize', 16)
    ylim([-2.0 1.0])
    xlim([-2.0 1.0])
    zlim([-1.5 1.5])
    zlabel('Re(\psi(x,y,t))', 'Fontsize', 18)
    xlabel('x', 'Fontsize', 18)
    ylabel('y', 'Fontsize', 18)
    view([19 41])

    main_title = "Im(\psi(x,y,t)) MLP Solutions for k = 3";

    subplot(1, 2, 2)
    surf(all_xpnts(:, :, k), all_ypnts(:, :, k), nn_pred_v(:, :, k),'DisplayName','Im(\psi(x,y,t)) MLP','LineWidth',2)
    shading interp;
    temp_title = sprintf('t = %.2f', times(k));
    title(temp_title, 'Fontsize', 16)
    ylim([-2.0 1.0])
    xlim([-2.0 1.0])
    zlim([-1.5 1.5])
    zlabel('Im(\psi(x,y,t))', 'Fontsize', 18)
    xlabel('x', 'Fontsize', 18)
    ylabel('y', 'Fontsize', 18)
    view([19 41])
    
    set(gcf, 'color', 'white'); 
    frame = getframe(fh);
    writeVideo(vidObj,frame);
    if k ~= N_ex
        clf
    end 
    
end

close(vidObj);

