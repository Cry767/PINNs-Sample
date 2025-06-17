classdef PINNTrainer < handle
    properties
        model       % 深層学習モデル（dlnetwork）
        lossValues  % 損失履歴
        optimizer   % オプティマイザ（SGDM など）
    end
    
    methods
        function obj = PINNTrainer(layers, learnRate)
            obj.model = dlnetwork(layers);
            obj.lossValues = [];
            obj.optimizer = trainingOptions("sgdm", ...
                "InitialLearnRate", learnRate, ...
                "MaxEpochs", 1, ...
                "MiniBatchSize", 1, ...
                "Shuffle", "never");
        end

        function obj = trainStep(obj, t_data, x_data, t_pinn, c, k)
            [loss, gradients] = dlfeval(@modelGradients, obj.model, t_data, x_data, t_pinn, c, k);
            
            % パラメータ更新
            obj.model = dlupdate(@(w, g) w - obj.optimizer.InitialLearnRate * g, obj.model, gradients);
        
            % 損失履歴の保存
            obj.lossValues(end+1) = extractdata(loss);
        end
    end
end
