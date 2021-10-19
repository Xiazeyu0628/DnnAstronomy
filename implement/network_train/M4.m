%% M4.m file

% try use the residential learning network replacing the mapping network

clear
clc



%% 
% This section has to be completed
input_size = 64;
filter_size = 64;
num_channels = 64;
lgraph= structure2(input_size,num_channels);
analyzeNetwork(lgraph);
%% 6. M3 implementation ...
% Use the functions provided above and your completed Unet architecture as
% appropriate

ratio = 0.01 ; 
% 输入是噪音图像，输出是噪声点
imds = imageDatastore('./project/data/noisy_images','ReadFcn',@NormalizeImageResize);
residual_imds = imageDatastore('./project/data/residual_images','ReadFcn',@NormalizeImageResize_residual);
nFiles = length(imds.Files);
RandIndices = randperm(nFiles);
ratio2num = round(ratio*nFiles);

validation_indices = RandIndices(1:ratio2num);
validation_set = subset(imds,validation_indices); 
validation_set_project = subset(residual_imds,validation_indices);


validation_combine=randomPatchExtractionDatastore(validation_set_project,validation_set,[64,64],...
     'DispatchInBackground',true,...
     'PatchesPerImage',24);
 
train_indices= RandIndices(ratio2num+1:nFiles);
train_set = subset(imds,train_indices); 
train_set_project = subset(residual_imds,train_indices);
train_combine=randomPatchExtractionDatastore(train_set_project,train_set,[64,64],...
     'DispatchInBackground',true,...
     'PatchesPerImage',24);

    
%% 7. M3 validation ...
options = trainingOptions('adam',...
    'MaxEpochs',20,...
    'MiniBatchSize',128,...
    'ValidationData',validation_combine,...
    'Verbose',true,...
    'Shuffle','every-epoch',...
    'InitialLearnRate',1e-2, ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.1,...
    'LearnRateDropPeriod',7,...
    'Plots','training-progress',...
    'GradientThreshold',1,...
    'ExecutionEnvironment','parallel');
    
    

[trainedNet,info] = trainNetwork(train_combine, lgraph, options);

cd('./project/net');
save M4_residual_0323 trainedNet;

