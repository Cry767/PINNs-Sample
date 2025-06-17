function obj = trainStep(obj, t_data, x_data, t_pinn, c, k)
    % t_data, x_data, t_pinn は dlarray として渡されることを前提とします
    % obj.model は dlnetwork オブジェクト
    % obj.lossFn は損失関数ハンドル（例: @mseLoss など）
    % obj.optimizer は optimizer オブジェクト（例: adamupdate 用）
    % obj.lossValues は損失履歴の配列

    % 学習用に微分追跡を開始
    dlXData = dlarray(x_data, 'CB');
    dlTData = dlarray(t_data, 'CB');
    dlTPinn = dlarray(t_pinn, 'CB');

    % 損失と勾配を計算
    [loss, gradients] = dlfeval(@modelGradients, obj.model, dlTData, dlXData, dlTPinn, c, k, obj.lossFn);

    % パラメータ更新
    [obj.model, obj.optimizer] = adamupdate(obj.model, gradients, obj.optimizer);

    % 損失を記録
    obj.lossValues(end+1) = extractdata(loss);
end

function [loss, gradients] = modelGradients(model, t_data, x_data, t_pinn, c, k, lossFn)
    % 観測データに対する予測
    x_pred = forward(model, t_data);
    loss1 = lossFn(x_pred, x_data);

    % PINNに基づく物理損失計算
    x_pred_pinn = forward(model, t_pinn);

    % 1階微分
    dx_dt = dlgradient(sum(x_pred_pinn, 'all'), t_pinn);

    % 2階微分
    dx_dt2 = dlgradient(sum(dx_dt, 'all'), t_pinn);

    % 物理モデルに基づく誤差
    loss_physics = dx_dt2 + c * dx_dt + k * x_pred_pinn;

    % loss2 は小さい重みでスケーリング
    lambda = 5.0*1e-2;
    loss2 = lambda * lossFn(loss_physics, zeros(size(loss_physics), 'like', loss_physics));

    % トータル損失
    loss = loss1 + loss2;

    % 勾配計算
    gradients = dlgradient(loss, model.Learnables);
end