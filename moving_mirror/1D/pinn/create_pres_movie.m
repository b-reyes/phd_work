

load 1d_matlab_movie_data_k_5.mat

main_title = "Solutions for k = 5";

vidObj = VideoWriter('1D_moving_mirror_movie.mp4', 'MPEG-4');
vidObj.FrameRate = 15;
open(vidObj);
fh = figure;

N_ex = 161;

for k=1:N_ex

    plot(all_xpnts(:, k), exact_u(:, k), 'r-','DisplayName','Re(\psi(x,t)) Exact','LineWidth',2)
    hold on
    plot(all_xpnts(:, k), nn_pred_u(:, k), 'b--','DisplayName','Re(\psi(x,t)) MLP','LineWidth',2)

    plot(all_xpnts(:, k), exact_v(:, k), 'k-','DisplayName','Im(\psi(x,t)) Exact','LineWidth',2)
    plot(all_xpnts(:, k), nn_pred_v(:, k), '--', 'color', [0 0.5 0], 'DisplayName','Im(\psi(x,t)) MLP','LineWidth',2)

    temp_title = sprintf('t = %.2f', times(k));
    title(temp_title, 'Fontsize', 16)
    ylim([-1.1 1.1])
    xlim([-2.0 1.0])
    ylabel('\psi(x,t)', 'Fontsize', 18)

    hold off
    Lgnd = legend('show','Fontsize',13,'Location','NorthEastOutside');
%     Lgnd.Position(1) = -0.15;
%     Lgnd.Position(2) = 0.712;
    
    set(gcf, 'color', 'white'); 
%     pause(0.1)
    frame = getframe(fh);
    writeVideo(vidObj,frame);
    if k ~= N_ex
        clf
    end 
    
end

close(vidObj);



