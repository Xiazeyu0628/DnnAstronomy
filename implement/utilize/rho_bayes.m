% use the bayes optimization machine to find best rho for hybird algorithm
net=load('-mat','./project/net/M3_0223.mat');
noise_level = 0.07;

optimVars = optimizableVariable('rho',[100,200],'Type','integer');

ObjFcn = rho_finder(net.trainedNet,noise_level);

BayesObject = bayesopt(ObjFcn,optimVars, ...
    'MaxTime',Inf, ...
    'MaxObjectiveEvaluations',30,...
    'IsObjectiveDeterministic',false, ...
    'NumSeedPoints',4,...
    'ExplorationRatio',0.5,...
    'Verbose',1);
 
bestIdx = BayesObject.IndexOfMinimumTrace(end);
fileName = BayesObject.UserDataTrace{bestIdx};
savedStruct = load(fileName);
valError = savedStruct.valError;