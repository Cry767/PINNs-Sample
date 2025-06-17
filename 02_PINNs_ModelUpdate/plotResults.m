function plotResults(model, t_true, x_true, t_obs, x_obs)
    t_true_dl = dlarray(t_true, 'CB');
    x_pred = extractdata(forward(model, t_true_dl));

    figure;
    plot(t_true, x_true, 'k-', 'LineWidth', 2); hold on;
    plot(t_true, x_pred, 'b--', 'LineWidth', 2);
    scatter(t_obs, x_obs, 80, 'r', 'filled');

    legend('解析解（真値）', 'PINNsによる予測', '観測点', 'Location', 'northeast');
    xlabel('時間 t'); ylabel('x(t)');
    title('PINNsによる予測と解析解の比較');
    grid on;
end
