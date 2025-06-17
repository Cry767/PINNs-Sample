% 必要なアドオン: Deep Learning Toolbox, parallel computing toolbox
% 参考アドレス
% https://techblog.insightedge.jp/entry/pinns-mass-spring-damper
clear;
% データ作成
t_data = linspace(0, 1, 5)';
t_pinn = linspace(0, 1, 30)';
c = 0.5;
k = 2.0;

% 減衰振動の模擬データ生成
omega_n = sqrt(k);
zeta = c / (2*omega_n);
omega_d = omega_n * sqrt(1 - zeta^2);
x_data = exp(-zeta*omega_n*t_data) .* cos(omega_d*t_data);

% dlarray に変換（'CB' → Channel, Batch）
t_data = dlarray(t_data', 'CB');
x_data = dlarray(x_data', 'CB');
t_pinn = dlarray(t_pinn', 'CB');

% t_data, x_data, t_pinn を GPU に移す
t_data = gpuArray(t_data);
x_data = gpuArray(x_data);
t_pinn = gpuArray(t_pinn);

% モデル定義（1入力 → 1隠れ層 → 1出力）
layers = [
    % featureInputLayer(1, "Normalization", "none", "Name", "input")
    % fullyConnectedLayer(20, "Name", "fc1")
    % tanhLayer("Name", "tanh1")
    % fullyConnectedLayer(1, "Name", "output")
    featureInputLayer(1, "Normalization", "none", "Name", "input")
    fullyConnectedLayer(20, "Name", "fc1")
    tanhLayer("Name", "tanh1")
    % fullyConnectedLayer(40, "Name", "fc2")
    % tanhLayer("Name", "tanh2")
    fullyConnectedLayer(1, "Name", "output")
];

% オブジェクト生成と1ステップ学習
trainer = PINNTrainer(layers, 1e-2);

numEpochs = 5000;
for i = 1:numEpochs
    trainer.trainStep(t_data, x_data, t_pinn, c, k);

    % 損失を定期的に表示
    if mod(i, 10) == 0
        disp("Epoch " + i + ": Loss = " + trainer.lossValues(end));
    end
    if abs(trainer.lossValues(end)) < 1e-3
        disp("収束したため終了");
        break;
    end
end

t_true = linspace(0, 10, 1000)';
x_analytical = generateAnalyticalSolution(c, k, t_true);

% 観測点を抽出（等間隔やランダム）
idx_obs = round(linspace(1, length(t_true), 10));
t_obs = t_true(idx_obs);
x_obs = x_analytical(idx_obs);

% 結果の出力
plotResults(trainer.model, t_data, x_data, t_obs, x_obs); %(model, t_true, x_true, t_obs, x_obs)