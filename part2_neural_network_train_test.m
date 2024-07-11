function [valid_acc, test_acc] = part2_neural_network_train_test( ...
    net, max_epochs, number_of_layers_to_freeze, ...
    init_lr, lr_schedule, lr_drop_factor, lr_drop_period, ...
    optimizer, miniBatchSize, ...
    train_ds, valid_ds, test_ds)


    % extract the image siz
    if isa(net, 'nnet.cnn.layer.Layer')
        inputSize = net(1).InputSize;
    else
        inputSize = net.Layers(1).InputSize;
    end
    
    % extract the layer graph from the trained network and input size defined by the network
    if ~isa(net,'nnet.cnn.LayerGraph')
        lgraph = layerGraph(net);
    else
        lgraph = net;
    end
    %if isa(net,'SeriesNetwork')
    %    lgraph = layerGraph(net.Layers);
    %else
    %    lgraph = layerGraph(net);
    %end
    
    % find layers to replace (two final layers corresponding to a certain dataset
    [learnableLayer,classLayer] = findLayersToReplace(lgraph);
    
    % replace last trainable layer to have the number of neurons = number of
    % classes 
    numClasses = numel(categories(train_ds.Labels));
    if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
     newLearnableLayer = fullyConnectedLayer(numClasses, ...
     'Name','new_fc', ...
     'WeightLearnRateFactor',10, ...
     'BiasLearnRateFactor',10);
    elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
     newLearnableLayer = convolution2dLayer(1,numClasses, ...
     'Name','new_conv', ...
     'WeightLearnRateFactor',10, ...
     'BiasLearnRateFactor',10);
    end
    lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);
    
    % replace the classification layer (deciding what the network ooutput is)
    % with the new one without classification labels. trainLabels automatically
    % sets them during training 
    newClassLayer = classificationLayer('Name','new_classoutput');
    lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);
    
    % freeze certain layers
    if number_of_layers_to_freeze > 0 
        layers = lgraph.Layers;
        connections = lgraph.Connections;
        layers(1:number_of_layers_to_freeze) = freezeWeights(layers(1:number_of_layers_to_freeze));
        lgraph = createLgraphUsingConnections(layers,connections);
    end
    
    % traind data resizing and augmentation 
    pixelRange = [-30 30];
    scaleRange = [0.9 1.1];
    imageAugmenter = imageDataAugmenter( ...
     'RandXReflection',true, ...
     'RandXTranslation',pixelRange, ...
     'RandYTranslation',pixelRange, ...
     'RandXScale',scaleRange, ...
     'RandYScale',scaleRange);
    augimdsTrain = augmentedImageDatastore(inputSize(1:2),train_ds, ...
     'DataAugmentation',imageAugmenter);
    
    % validation data resizing
    augimdsValidation = augmentedImageDatastore(inputSize(1:2),valid_ds);
    
    % test data resizing
    augimdsTest = augmentedImageDatastore(inputSize(1:2),test_ds);
    
    % train config
    valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
    options = trainingOptions(optimizer, ...
     'MiniBatchSize',miniBatchSize, ...
     'MaxEpochs',max_epochs, ...
     'InitialLearnRate',init_lr, ...
     'LearnRateSchedule', lr_schedule, ...
     'LearnRateDropFactor', lr_drop_factor, ...
     'LearnRateDropPeriod', lr_drop_period, ...
     'Shuffle','every-epoch', ...
     'ValidationData',augimdsValidation, ...
     'ValidationFrequency',valFrequency, ...
     'Verbose',false, ...
     'Plots','training-progress');
    
    % train the network 
    net = trainNetwork(augimdsTrain,lgraph,options);
    analyzeNetwork(lgraph)

    % predict on validation set 
    [YPred,~] = classify(net,augimdsValidation);
    valid_acc = mean(YPred == valid_ds.Labels);
    
    % predict on test set 
    [YPred,~] = classify(net,augimdsTest);
    test_acc = mean(YPred == test_ds.Labels);
end