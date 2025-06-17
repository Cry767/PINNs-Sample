function [loss, gradients] = modelGradients(model, t_data, x_data, t_pinn, c, k)
    % データ誤差
    x_pred = forward(model, t_data);
    loss1 = mean((x_pred - x_data).^2, 'all');

    % PINN 誤差用の勾配計算
    numPoints = size(t_pinn, 2);
    dx_dt  = zeros(1, numPoints, 'like', t_pinn);
    dx_dt2 = zeros(1, numPoints, 'like', t_pinn);
    x_pred_pinn = zeros(1, numPoints, 'like', t_pinn);

    for i = 1:numPoints
        ti = t_pinn(:, i);
        ti = dlarray(ti, 'CB');

        % Forward & 微分処理
        xi = forward(model, ti);
        dxi = dlgradient(sum(xi), ti, 'EnableHigherDerivatives', true);
        dxi2 = dlgradient(sum(dxi), ti);

        dx_dt(i) = dxi;
        dx_dt2(i) = dxi2;
        x_pred_pinn(i) = xi;
    end

    loss_physics = dx_dt2 + c * dx_dt + k * x_pred_pinn;
    loss2 = 5.0*1e-4 * mean((loss_physics).^2, 'all');

    % 合算損失
    loss = loss1 + loss2;

    % 勾配計算
    gradients = dlgradient(loss, model.Learnables);
end
