function [] = Action_Recognition_DL_from_scratch()

    trainset_path = fullfile("data/", "Dataset", "TrainSet");
    train_ds = imageDatastore(trainset_path, 'IncludeSubfolders',true, 'LabelSource','foldernames');
    
    testset_path = fullfile("data/", "Dataset", "TestSet");
    test_ds = imageDatastore(testset_path, 'IncludeSubfolders',true, 'LabelSource','foldernames');
    
    %% split training into train and validaation sets
    [train_ds,valid_ds] = splitEachLabel(train_ds,0.7);
    
    %% define custom networks input size 
    input_size = [256 256 3];
    num_classes = 7;
    
    %% one_layer_nn
    one_layer_nn = [
        imageInputLayer(input_size)
        convolution2dLayer(5,20)
        batchNormalizationLayer
        reluLayer
        fullyConnectedLayer(num_classes)
        softmaxLayer
        classificationLayer];
    
    hyperparams = {
        {10, 0, 3e-4, 'none', 0, 1, 'sgdm', 10}, ...
        {10, 0, 0.001, 'none', 0, 1, 'sgdm', 10}, ...
        {10, 0, 3e-4, 'none', 0, 1, 'adam', 10}, ...
        {10, 0, 0.001, 'none', 0, 1, 'adam', 10}, ...
        };
    
    [one_layer_nn_best_valid_acc, one_layer_nn_best_test_acc, one_layer_nn_best_hyper_params] = part2_tune_hyperparams(one_layer_nn, hyperparams, train_ds, valid_ds, test_ds);
    %% five_layers_nn
    
    five_layers_nn = [
        imageInputLayer(input_size)
        
        convolution2dLayer(3,8,'Padding','same')
        batchNormalizationLayer
        reluLayer
        
        maxPooling2dLayer(2,'Stride',2)
        
        convolution2dLayer(3,16,'Padding','same')
        batchNormalizationLayer
        reluLayer
        
        maxPooling2dLayer(2,'Stride',2)
        
        convolution2dLayer(3,32,'Padding','same')
        batchNormalizationLayer
        reluLayer
    
        maxPooling2dLayer(2,'Stride',2)
    
        convolution2dLayer(3,64,'Padding','same')
        batchNormalizationLayer
        reluLayer
    
        maxPooling2dLayer(2,'Stride',2)
    
        convolution2dLayer(3,128,'Padding','same')
        batchNormalizationLayer
        reluLayer
    
        fullyConnectedLayer(num_classes)
        softmaxLayer
        classificationLayer];
    
    hyperparams = {
        {30, 0, 3e-4, 'none', 0, 1, 'sgdm', 10}, ...
        {30, 0, 0.001, 'none', 0, 1, 'sgdm', 10}, ...
        {30, 0, 3e-4, 'none', 0, 1, 'adam', 10}, ...
        {30, 0, 0.001, 'none', 0, 1, 'adam', 10}, ...
        {30, 0, 3e-4, 'piecewise', 0.5, 5, 'sgdm', 10}, ...
        {30, 0, 0.001, 'piecewise', 0.5, 5, 'adam', 10}, ...
        };
    
    [five_layers_nn_best_valid_acc, five_layers_nn_best_test_acc, five_layers_nn_best_hyper_params] = part2_tune_hyperparams(five_layers_nn, hyperparams, train_ds, valid_ds, test_ds);
    %% double_five_layers_nn
    
    double_five_layers_nn = [
        imageInputLayer(input_size)
        
        convolution2dLayer(3,8,'Padding','same')
        batchNormalizationLayer
        reluLayer
        
        convolution2dLayer(3,8,'Padding','same')
        batchNormalizationLayer
        reluLayer
        
        maxPooling2dLayer(2,'Stride',2)
        
        convolution2dLayer(3,16,'Padding','same')
        batchNormalizationLayer
        reluLayer
    
        convolution2dLayer(3,16,'Padding','same')
        batchNormalizationLayer
        reluLayer
    
        maxPooling2dLayer(2,'Stride',2)
    
        convolution2dLayer(3,64,'Padding','same')
        batchNormalizationLayer
        reluLayer
    
        convolution2dLayer(3,64,'Padding','same')
        batchNormalizationLayer
        reluLayer
    
        maxPooling2dLayer(2,'Stride',2)
    
        convolution2dLayer(3,128,'Padding','same')
        batchNormalizationLayer
        reluLayer
    
        convolution2dLayer(3,128,'Padding','same')
        batchNormalizationLayer
        reluLayer
    
        maxPooling2dLayer(2,'Stride',2)
    
        convolution2dLayer(3,256,'Padding','same')
        batchNormalizationLayer
        reluLayer
    
        convolution2dLayer(3,256,'Padding','same')
        batchNormalizationLayer
        reluLayer
    
        fullyConnectedLayer(64)
        fullyConnectedLayer(num_classes)
        softmaxLayer
        classificationLayer];
    
    
    hyperparams = {
        {20, 0, 3e-4, 'none', 0, 1, 'sgdm', 10}, ...
        {20, 0, 0.001, 'none', 0, 1, 'sgdm', 10}, ...
        {20, 0, 3e-4, 'none', 0, 1, 'adam', 10}, ...
        {20, 0, 0.001, 'none', 0, 1, 'adam', 10}, ...
        {20, 0, 3e-4, 'piecewise', 0.5, 5, 'sgdm', 10}, ...
        {20, 0, 0.001, 'piecewise', 0.5, 5, 'adam', 10}, ...
        };
    
    
    [double_five_layers_nn_best_valid_acc, double_five_layers_nn_best_test_acc, double_five_layers_nn_best_hyper_params] = part2_tune_hyperparams(double_five_layers_nn, hyperparams, train_ds, valid_ds, test_ds);

end
