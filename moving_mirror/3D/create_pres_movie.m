close all 
clear all

load /Users/brandonreyes/Desktop/3D_moving_mirror_movie_data/3d_matlab_movie_data_k_1.mat
% load 3d_matlab_movie_data_k_1.mat
acon_z = 0.25;

vidObj = VideoWriter('3D_moving_mirror_movie.mp4', 'MPEG-4');
vidObj.FrameRate = 15;
open(vidObj);
fh = figure;

N_ex = 101;

for k=1:N_ex

    subplot(1, 2, 1)
    [X, Y, Z] = meshgrid(all_xpnts(:, k), all_ypnts(:, k), all_zpnts(:, k));
    [xslice, yslice] = meshgrid(all_xpnts(:, k), all_ypnts(:, k));
    zslice = xslice; %(xslice + yslice)/2;
    slice(X, Y, Z, nn_pred_u(:, :, :, k), xslice, yslice, zslice)    % display the slices
    shading interp;
    hold on
    slice(X, Y, Z, nn_pred_u(:, :, :, k),[],[],(-2.0 + acon_z*times(k)^2)/2)
    shading interp;
    hold off
    temp_title = ['Re(\psi(x,y,z,t)) at ', sprintf('t = %.2f', times(k))];
    title(temp_title, 'Fontsize', 16)
    xlim([-2 1.1])
    ylim([-2 1.1])
    zlim([-2 1.1])
    xlabel('x', 'Fontsize', 18)
    ylabel('y', 'Fontsize', 18)
    zlabel('z', 'Fontsize', 18)
    view(-34,24)

    subplot(1, 2, 2)
    [X, Y, Z] = meshgrid(all_xpnts(:, k), all_ypnts(:, k), all_zpnts(:, k));
    [xslice, yslice] = meshgrid(all_xpnts(:, k), all_ypnts(:, k));
    zslice = xslice; %(xslice + yslice)/2;
    slice(X, Y, Z, nn_pred_v(:, :, :, k), xslice, yslice, zslice)    % display the slices
    shading interp;
    hold on
    slice(X, Y, Z, nn_pred_v(:, :, :, k),[],[],(-2.0 + acon_z*times(k)^2)/2)
    shading interp;
    hold off
    temp_title = ['Im(\psi(x,y,z,t)) at ', sprintf('t = %.2f', times(k))];
    title(temp_title, 'Fontsize', 16)
    xlim([-2 1.1])
    ylim([-2 1.1])
    zlim([-2 1.1])
    xlabel('x', 'Fontsize', 18)
    ylabel('y', 'Fontsize', 18)
    zlabel('z', 'Fontsize', 18)
    view(-34,24)
    
    set(gcf, 'color', 'white'); 
    frame = getframe(fh);
    writeVideo(vidObj,frame);
    if k ~= N_ex
        clf
    end 
    
end

close(vidObj);




