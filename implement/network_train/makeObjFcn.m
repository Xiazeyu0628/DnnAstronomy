function ObjFcn = makeObjFcn(train_set_project,train_set,validation_set_project,validation_set)

ObjFcn = @valErrorFun;

function [reconstruction_result,con,fileName] = valErrorFun(optVars)
    %%data process
    validation_combine=randomPatchExtractionDatastore(validation_set_project,validation_set,[32,32],...
     'DispatchInBackground',true,...
     'PatchesPerImage',24);
 
     train_combine=randomPatchExtractionDatastore(train_set_project,train_set,[32,32],...
     'DispatchInBackground',true,...
     'PatchesPerImage',24);
 
%     %%layer
%     input_size = 32;
%     num_channels = 32;
%     layers = [
%     imageInputLayer([input_size input_size 1], 'Name','Input')
% 
%     convolution2dLayer(3,1,'NumChannels',1,'Padding','same','Name','conv11')
%     batchNormalizationLayer('Name','bn11')
%     reluLayer('Name','relu11')
%     
%     convolution2dLayer(3,num_channels,'NumChannels',num_channels,'Padding','same','Name','conv12')
%     batchNormalizationLayer('Name','bn12')
%     reluLayer('Name','relu12')
%     
%      convolution2dLayer(3,num_channels,'NumChannels',num_channels,'Padding','same','Name','conv13')
%     batchNormalizationLayer('Name','bn13')
%     reluLayer('Name','relu13')
%     
%      convolution2dLayer(3,num_channels,'NumChannels',num_channels,'Padding','same','Name','conv14')
%     batchNormalizationLayer('Name','bn14')
%     reluLayer('Name','relu14')
%     
%      convolution2dLayer(3,num_channels,'NumChannels',num_channels,'Padding','same','Name','conv15')
%     batchNormalizationLayer('Name','bn15')
%     reluLayer('Name','relu15')
%     
%      convolution2dLayer(3,num_channels,'NumChannels',num_channels,'Padding','same','Name','conv16')
%     batchNormalizationLayer('Name','bn16')
%     reluLayer('Name','relu16')
%     
%      convolution2dLayer(3,num_channels,'NumChannels',num_channels,'Padding','same','Name','conv17')
%     batchNormalizationLayer('Name','bn17')
%     reluLayer('Name','relu17')
%     
%      convolution2dLayer(3,num_channels,'NumChannels',num_channels,'Padding','same','Name','conv18')
%     batchNormalizationLayer('Name','bn18')
%     reluLayer('Name','relu18')
%     
%      convolution2dLayer(3,num_channels,'NumChannels',num_channels,'Padding','same','Name','conv19')
%     batchNormalizationLayer('Name','bn19')
%     reluLayer('Name','relu19')
%     
%      convolution2dLayer(3,num_channels,'NumChannels',num_channels,'Padding','same','Name','conv110')
%     batchNormalizationLayer('Name','bn110')
%     reluLayer('Name','relu110')
%     
%      convolution2dLayer(3,num_channels,'NumChannels',num_channels,'Padding','same','Name','conv111')
%     batchNormalizationLayer('Name','bn111')
%     reluLayer('Name','relu111')
%     
%      convolution2dLayer(3,num_channels,'NumChannels',num_channels,'Padding','same','Name','conv112')
%     batchNormalizationLayer('Name','bn112')
%     reluLayer('Name','relu112')
%     
%      convolution2dLayer(3,num_channels,'NumChannels',num_channels,'Padding','same','Name','conv113')
%     batchNormalizationLayer('Name','bn113')
%     reluLayer('Name','relu113')
%     
%      convolution2dLayer(3,num_channels,'NumChannels',num_channels,'Padding','same','Name','conv114')
%     batchNormalizationLayer('Name','bn114')
%     reluLayer('Name','relu114')
%     
%      convolution2dLayer(3,num_channels,'NumChannels',num_channels,'Padding','same','Name','conv115')
%     batchNormalizationLayer('Name','bn115')
%     reluLayer('Name','relu115')
%     
%      convolution2dLayer(3,num_channels,'NumChannels',num_channels,'Padding','same','Name','conv116')
%     batchNormalizationLayer('Name','bn116')
%     reluLayer('Name','relu116')
%     
%     convolution2dLayer(3,1,'NumChannels',1,'Padding','same','Name','conv117')
%     batchNormalizationLayer('Name','bn117')
%     reluLayer('Name','relu117')
%     
%     additionLayer(2,'Name','add_end')
%     
%     regressionLayer('Name','output')
%     
% ];
%  lgraph = layerGraph(layers);
%  lgraph = connectLayers(lgraph,'Input','add_end/in2');
input_size = 32;
filter_size = 64;
num_channels = 64;
layers = [
    imageInputLayer([input_size input_size 1], 'Name','Input')

    convolution2dLayer(3,filter_size,'NumChannels',1,'Padding','same','Name','conv11')
    reluLayer('Name','relu11')
    
    convolution2dLayer(3,filter_size,'NumChannels',num_channels,'Padding','same','Name','conv12')
    batchNormalizationLayer('Name','bn12')
    reluLayer('Name','relu12')
    
     convolution2dLayer(3,filter_size,'NumChannels',num_channels,'Padding','same','Name','conv13')
    batchNormalizationLayer('Name','bn13')
    reluLayer('Name','relu13')
    
     convolution2dLayer(3,filter_size,'NumChannels',num_channels,'Padding','same','Name','conv14')
    batchNormalizationLayer('Name','bn14')
    reluLayer('Name','relu14')
    
     convolution2dLayer(3,filter_size,'NumChannels',num_channels,'Padding','same','Name','conv15')
    batchNormalizationLayer('Name','bn15')
    reluLayer('Name','relu15')
    
     convolution2dLayer(3,filter_size,'NumChannels',num_channels,'Padding','same','Name','conv16')
    batchNormalizationLayer('Name','bn16')
    reluLayer('Name','relu16')
    
     convolution2dLayer(3,filter_size,'NumChannels',num_channels,'Padding','same','Name','conv17')
    batchNormalizationLayer('Name','bn17')
    reluLayer('Name','relu17')
    
     convolution2dLayer(3,filter_size,'NumChannels',num_channels,'Padding','same','Name','conv18')
    batchNormalizationLayer('Name','bn18')
    reluLayer('Name','relu18')
    
     convolution2dLayer(3,filter_size,'NumChannels',num_channels,'Padding','same','Name','conv19')
    batchNormalizationLayer('Name','bn19')
    reluLayer('Name','relu19')
    
     convolution2dLayer(3,filter_size,'NumChannels',num_channels,'Padding','same','Name','conv110')
    batchNormalizationLayer('Name','bn110')
    reluLayer('Name','relu110')
    
     convolution2dLayer(3,filter_size,'NumChannels',num_channels,'Padding','same','Name','conv111')
    batchNormalizationLayer('Name','bn111')
    reluLayer('Name','relu111')
    
     convolution2dLayer(3,filter_size,'NumChannels',num_channels,'Padding','same','Name','conv112')
    batchNormalizationLayer('Name','bn112')
    reluLayer('Name','relu112')
    
     convolution2dLayer(3,filter_size,'NumChannels',num_channels,'Padding','same','Name','conv113')
    batchNormalizationLayer('Name','bn113')
    reluLayer('Name','relu113')
    
     convolution2dLayer(3,filter_size,'NumChannels',num_channels,'Padding','same','Name','conv114')
    batchNormalizationLayer('Name','bn114')
    reluLayer('Name','relu114')
    
     convolution2dLayer(3,filter_size,'NumChannels',num_channels,'Padding','same','Name','conv115')
    batchNormalizationLayer('Name','bn115')
    reluLayer('Name','relu115')
    
     convolution2dLayer(3,filter_size,'NumChannels',num_channels,'Padding','same','Name','conv116')
    batchNormalizationLayer('Name','bn116')
    reluLayer('Name','relu116')
    
    convolution2dLayer(3,1,'NumChannels',num_channels,'Padding','same','Name','conv117')
    

    
    regressionLayer('Name','output')
    
];
 lgraph = layerGraph(layers);

options = trainingOptions('adam',...
    'MaxEpochs',optVars.MaxEpochs,...
    'MiniBatchSize',128,...
    'ValidationData',validation_combine,...
    'ValidationPatience',10,...
    'Verbose',true,...
    'Shuffle','every-epoch',...
    'InitialLearnRate',1e-3, ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.1,...
    'LearnRateDropPeriod',optVars.LearnRateDropPeriod,...
    'Plots','training-progress',...
    'GradientThreshold',1,...
    'ExecutionEnvironment','parallel');

     trainedNet = trainNetwork(train_combine,lgraph,options);
     reconstruction_result = -1*(reconstruction_cal(trainedNet));
     fileName =['./project/net/bayes/',num2str(reconstruction_result),'.mat'];
     save(fileName,'trainedNet','reconstruction_result','options')
     con = []
end
end

