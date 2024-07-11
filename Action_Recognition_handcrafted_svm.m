function [] = Action_Recognition_handcrafted_svm()

    %% laod train and test datasets
    trainset_path = fullfile("data/", "Dataset", "TrainSet");
    train_ds = imageDatastore(trainset_path, 'IncludeSubfolders',true, 'LabelSource','foldernames');
    
    testset_path = fullfile("data/", "Dataset", "TestSet");
    test_ds = imageDatastore(testset_path, 'IncludeSubfolders',true, 'LabelSource','foldernames');
    
    %% visualize sample images
    visualize_imgs(train_ds, test_ds);
    
    %% feature extraction
    
    % hog/lbp feature extraction
    cell_size = [16 16]; % [8 8], [4 4] 
    % feature_extractor = @(ds) part1_lbp_fe(ds, cell_size);
    feature_extractor = @(ds) part1_hog_fe(ds, cell_size);
    
    train_features = feature_extractor(train_ds);
    train_labels = train_ds.Labels;
    
    test_features = feature_extractor(test_ds);
    test_labels = test_ds.Labels;
    
    
    %% training & hyperparams tunning with coss-validation
    c = cvpartition(numel(train_ds.Files),'KFold',5);
    opts = struct('CVPartition',c,'AcquisitionFunctionName','expected-improvement-plus');
    t = templateSVM('Standardize',true,'KernelFunction','polynomial');
    model = fitcecoc(train_features,train_labels,'Learners',t, 'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts);
    % testing
    pred_labels = predict(model, test_features);
    acc = mean(pred_labels == test_labels);
    disp(acc);

end
