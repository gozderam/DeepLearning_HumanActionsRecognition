function [] = Action_Recognition_handcrafted_nn()

    %% laod train and test datasets
    trainset_path = fullfile("data/", "Dataset", "TrainSet");
    train_ds = imageDatastore(trainset_path, 'IncludeSubfolders',true, 'LabelSource','foldernames');
    
    testset_path = fullfile("data/", "Dataset", "TestSet");
    test_ds = imageDatastore(testset_path, 'IncludeSubfolders',true, 'LabelSource','foldernames');
    
    %% visualize sample images
    visualize_imgs(train_ds, test_ds);
    
    %% feature extraction
    
    % hog feature extraction
    cell_size = [16 16]; % [8 8], [4 4] 
    %feature_extractor = @(ds) part1_lbp_fe(ds, cell_size);
    feature_extractor = @(ds) part1_hog_fe(ds, cell_size);
    
    [train_ds,valid_ds] = splitEachLabel(train_ds,0.7);
    
    train_features = feature_extractor(train_ds);
    train_labels = train_ds.Labels;
    
    valid_features = feature_extractor(valid_ds);
    valid_labels = valid_ds.Labels;
    
    test_features = feature_extractor(test_ds);
    test_labels = test_ds.Labels;
    
    num_classes = 7;
    
    hyperparams = {
        {10, 3e-4, 'none', 0, 1, 'sgdm', 10}, ...
        {10, 0.001, 'none', 0, 1, 'sgdm', 10}, ...
        {10, 3e-4, 'none', 0, 1, 'adam', 10}, ...
        {10, 0.001, 'none', 0, 1, 'adam', 10}, ...
        };
    
    %% networks definitions, training and testing for one_h_layer_nn
    
    one_h_layer_nn = [featureInputLayer(size(train_features, 2),'Normalization', 'zscore')
        fullyConnectedLayer(128)
        reluLayer
        fullyConnectedLayer(num_classes)
        softmaxLayer
        classificationLayer];
    
    [one_h_layer_nn_best_valid_acc, one_h_layer_nn_best_test_acc, one_h_layer_nn_best_hyper_params] ...
        = part1_tune_hyperparams(one_h_layer_nn, hyperparams, train_features, train_labels, valid_features, valid_labels, test_features, test_labels);
    
    
    %% networks definitions, training and testing for two_h_layers_nn
    
    two_h_layers_nn = [featureInputLayer(size(train_features, 2),'Normalization', 'zscore')
        fullyConnectedLayer(64)
        reluLayer
        fullyConnectedLayer(128)
        reluLayer
        fullyConnectedLayer(num_classes)
        softmaxLayer
        classificationLayer];
    
    [two_h_layers_nn_best_valid_acc, two_h_layers_nn_best_test_acc, two_h_layers_nn_best_hyper_params] ...
        = part1_tune_hyperparams(two_h_layers_nn, hyperparams, train_features, train_labels, valid_features, valid_labels, test_features, test_labels);

end