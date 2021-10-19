%% M3.m file
%
% %
% This file is for your implementation and validation of M3 (sections 6 and 7).
% It contains useful functions (sections 1 to 4) as well as a partial
% implementation of the DnCNN for you to finalise (section 5).  
clear
clc


%% 4. Working with image patches
% The training of a denoiser is eased when using large number of images
% (batch size) and small image size (patches). We suggest that you train
% your network on a large number (>>1000) of patches of size 64x64, created
% from the original 1000 groundtruth images of size 256x256 in the training
% data set. The network will then act as a denoiser on 64x64 images.
% Your problem is to validate the method on the 256x256 images of the
% testing data set. The following function will allow you to
% apply your network to images of any size, by simply denoising 64x64 patches
% separately with the network.


%% 5. DnCNN architecture and implementation
% This section has to be completed
input_size = 64;
num_channels = 32;
lgraph = structure1(input_size,num_channels);
analyzeNetwork(lgraph);
 
%% 6. M3 implementation ...
% Use the functions provided above and your completed Unet architecture as
% appropriate

ratio = 0.01 ; 
imds = imageDatastore('./project/data/groundtruth_png/good','ReadFcn',@NormalizeImageResize);
imds2 = imageDatastore('./project/data/groundtruth_png/good','ReadFcn',@NormalizeImageResize_addnoise);
nFiles = length(imds.Files);
RandIndices = randperm(nFiles);
ratio2num = round(ratio*nFiles);

validation_indices = RandIndices(1:ratio2num);
validation_set = subset(imds,validation_indices); 
validation_set_project = subset(imds2,validation_indices);
validation_combine=randomPatchExtractionDatastore(validation_set_project,validation_set,[64,64],...
     'DispatchInBackground',true,...
     'PatchesPerImage',16);
 
train_indices= RandIndices(ratio2num+1:nFiles);
train_set = subset(imds,train_indices); 
train_set_project = subset(imds2,train_indices);
train_combine=randomPatchExtractionDatastore(train_set_project,train_set,[64,64],...
     'DispatchInBackground',true,...
     'PatchesPerImage',16);

    
%% 7. M3 validation ...
options = trainingOptions('adam',...
    'MaxEpochs',6,...
    'MiniBatchSize',128,...
    'ValidationData',validation_combine,...
    'Verbose',true,...
    'Shuffle','every-epoch',...
    'InitialLearnRate',1e-3, ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.1,...
    'LearnRateDropPeriod',3,...
    'Plots','training-progress',...
    'GradientThreshold',1,...
    'ExecutionEnvironment','cpu',...
    'CheckpointPath','./project/net/checkpointpath');
    
    

[trainedNet,info] = trainNetwork(train_combine, lgraph, options);

cd('./project/net');
save M3_0222 trainedNet;

