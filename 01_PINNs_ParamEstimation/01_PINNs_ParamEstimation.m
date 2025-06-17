
%==========================================================================
% PINN for parameter estimation in 1-DOF mass-spring-damper system using MATLAB
% Equation: m*x'' + c*x' + k*x = F(t)
% Goal: Estimate parameters m, c, k using measured F(t) and x(t)
%==========================================================================

% Clear environment
clear; clc; close all;

%----------------------------
% Generate synthetic data
%----------------------------
t = linspace(0, 10, 1000)';
dt = t(2) - t(1);
true_m = 1.0;
true_c = 0.5;
true_k = 2.0;
F = sin(2*pi*0.5*t);  % External force
x = zeros(size(t));
v = zeros(size(t));
a = zeros(size(t));
x(1) = 0; v(1) = 0;

% Simulate system using forward Euler
for i = 1:length(t)-1
    a(i) = (F(i) - true_c*v(i) - true_k*x(i)) / true_m;
    v(i+1) = v(i) + a(i)*dt;
    x(i+1) = x(i) + v(i)*dt;
end
a(end) = (F(end) - true_c*v(end) - true_k*x(end)) / true_m;

% Add small noise to x
x_noisy = x + 0.01*randn(size(x));

%----------------------------
% Define and train neural network
%----------------------------
inputs = [t, F];
targets = x_noisy;

% Define shallow network
net = fitnet(20);  % 20 hidden neurons
net.trainFcn = 'trainlm';  % Levenberg-Marquardt

% Train the network to fit x(t)
net = train(net, inputs', targets');

% Predict output and compute derivatives
x_pred = net(inputs')';
dxdt = gradient(x_pred, dt);
d2xdt2 = gradient(dxdt, dt);

%----------------------------
% Estimate parameters m, c, k by minimizing residual
%----------------------------
residual = @(p) d2xdt2 .* p(1) + dxdt .* p(2) + x_pred .* p(3) - F;
loss_fn = @(p) mean(residual(p).^2);

% Initial guesses
p0 = [0.8, 0.3, 1.5];

% Optimization
opts = optimoptions('fminunc','Algorithm','quasi-newton','Display','iter');
[param_est,fval] = fminunc(loss_fn, p0, opts);

% Display result
fprintf('Estimated parameters: m = %.4f, c = %.4f, k = %.4f\n', ...
        param_est(1), param_est(2), param_est(3));
fprintf('True parameters:      m = %.4f, c = %.4f, k = %.4f\n', ...
        true_m, true_c, true_k);
