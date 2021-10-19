clear
clc
addpath('utils/');
addpath('utils/lib/');
run('utils/lib/irt/setup.m');
ratio = 0.01 ; 
% imds = imageDatastore('./project/data/groundtruth_png/good','ReadFcn',@NormalizeImageResize);
% imds2 = imageDatastore('./project/data/groundtruth_png/good','ReadFcn',@NormalizeImageResize_addnoise);
imds = imageDatastore('./project/data/noisy_images','ReadFcn',@NormalizeImageResize);
imds2 = imageDatastore('./project/data/residual_image','ReadFcn',@NormalizeImageResize);
nFiles = length(imds.Files);
RandIndices = randperm(nFiles);
ratio2num = round(ratio*nFiles);

validation_indices = RandIndices(1:ratio2num);
validation_set = subset(imds,validation_indices); 
validation_set_project = subset(imds2,validation_indices);

train_indices= RandIndices(ratio2num+1:nFiles);
train_set = subset(imds,train_indices); 
train_set_project = subset(imds2,train_indices);

optimVars = [
    optimizableVariable('MaxEpochs',[6,20],'Type','integer')
    optimizableVariable('LearnRateDropPeriod',[2,10],'Type','integer')];

ObjFcn = makeObjFcn(train_set_project,train_set,validation_set_project,validation_set);

BayesObject = bayesopt(ObjFcn,optimVars, ...
    'MaxTime',Inf, ...
    'MaxObjectiveEvaluations',30,...
    'IsObjectiveDeterministic',false, ...
    'AcquisitionFunctionName','expected-improvement-plus',...
    'NumSeedPoints',4,...
    'ExplorationRatio',0.5,...
     'Verbose',1,...
     'UseParallel',false);

bestIdx = BayesObject.IndexOfMinimumTrace(end);
fileName = BayesObject.UserDataTrace{bestIdx};
savedStruct = load(fileName);
reconstruction_result = savedStruct.reconstruction_result;