function ObjFcn = rho_finder(trainedNet,noise_level)

ObjFcn = @valErrorFun;

function [reconstruction_result,con,fileName] = valErrorFun(optVars)
     
Nx = 512;
Ny = 512;
nlevel=2;
wv='db8';
dwtmode('per');
M = 30000;

imds = imageDatastore('./project/data/samples');
operators = load('-mat','./project/implement/simulated_data/operators/data.mat');

% parameters
options.verbose = 1;
options.rel_tol = 1e-5;
options.rel_tol2 = 1e-5;
options.delta = 1;
options.max_iter = 2000;
options.rho = optVars.rho;  % 惩罚项
options.delta = 1e-7;% 斜率，step size for the proximal gradient update 需要小于lipschitz continuous gradient

epsilon = noise_level*sqrt(M + 2*sqrt(M));

% Definition of the measurement operators Phit and Phi
Phi_t = @(x) HS_forward_operator(x,operators.Gw,operators.A); % forward measurement operator
Phi = @(y) HS_adjoint_operator(y,operators.Gw,operators.At,Nx,Ny); %associated adjoint operator

  i=1; %figure
result_matrix = zeros(2,size(imds.Files,1));

while hasdata(imds)
    
    % read and compare
    im = read(imds);
    im=im2double(im);
    im=im/max(max(im));
    y = Phi_t(im);
    x_project = Phi(y);
    x_project=x_project/max(max(x_project));


%% 2. Sparsity operators    
    % Definition of the sparsity operators Psi and Psit
    [alphad,S]=wavedec2(x_project,nlevel,wv);
    Psit = @(x) wavedec2(x, nlevel,wv); 
    Psi = @(x) waverec2(x,S,wv);

%% 3. noisy operators      
    % add noise operator
    add_noise = @(y) y + (randn(size(y)) + 1i*randn(size(y)))*noise_level/sqrt(2);

    % add noise and define eplison
    y_noise=add_noise(y);
    
%% 4. solve the value

    %admm
    %[xsol, rsnr, niter] = admm(y_noise,epsilon,Phi_t,Phi,Psi,Psit,options,net,im);
    [~, rsnr, ~] = admm_hybird(y_noise,epsilon,Phi_t,Phi,Psi,Psit,options,trainedNet,im);
    
   % write
   result_matrix(1,i) = i ;
   result_matrix(2,i) = rsnr ;
   break;
   i=i+1;
end 
rho_value = optVars.rho;
reconstruction_result = 1/mean(result_matrix(2,:));
fileName =['./project/parameters/rho_bayes/',num2str(reconstruction_result),'.mat'];
save(fileName,'rho_value','reconstruction_result')
con = []
end
end


