function [best_valid_acc, best_test_acc, best_hyper_params] = part2_tune_hyperparams( ...
    net, hyperparams, train_ds, valid_ds, test_ds)

    best_hyper_params = 0;
    best_valid_acc = 0;
    best_test_acc = 0;
    
    for i=1:size(hyperparams, 2)
        [valid_acc, test_acc] = part2_neural_network_train_test( ...
            net,  ...     % net
            hyperparams{i}{1}, ... % max_epochs
            hyperparams{i}{2}, ... % number_of_layers_to_freeze
            hyperparams{i}{3}, ... % init_lr
            hyperparams{i}{4}, ... % lr_schedule
            hyperparams{i}{5}, ... % lr_drop_factor
            hyperparams{i}{6}, ... % lr_drop_period
            hyperparams{i}{7}, ... % optimizer
            hyperparams{i}{8}, ... % miniBatchSize
            train_ds, valid_ds, test_ds);
    
        if valid_acc > best_valid_acc
            best_valid_acc = valid_acc;
            best_test_acc = test_acc;
            best_hyper_params = hyperparams{i};
        end
        
    end
end