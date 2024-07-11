function [best_valid_acc, best_test_acc, best_hyper_params] = part1_tune_hyperparams( ...
    net, hyperparams, train_f, train_l, valid_f, valid_l, test_f, test_l)

    best_hyper_params = 0;
    best_valid_acc = 0;
    best_test_acc = 0;
    
    for i=1:size(hyperparams, 2)
        [valid_acc, test_acc] = part1_neural_network_train_test( ...
            net,  ...     % net
            hyperparams{i}{1}, ... % max_epochs
            hyperparams{i}{2}, ... % init_lr
            hyperparams{i}{3}, ... % lr_schedule
            hyperparams{i}{4}, ... % lr_drop_factor
            hyperparams{i}{5}, ... % lr_drop_period
            hyperparams{i}{6}, ... % optimizer
            hyperparams{i}{7}, ... % miniBatchSize
            train_f, train_l, valid_f, valid_l, test_f, test_l);
    
        if valid_acc > best_valid_acc
            best_valid_acc = valid_acc;
            best_test_acc = test_acc;
            best_hyper_params = hyperparams{i};
        end
        
    end
end