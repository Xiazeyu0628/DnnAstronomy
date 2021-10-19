clear 
clc

addpath('utils/');
addpath('utils/lib/');
%run('utils/lib/irt/setup.m');

% constants
Nx = 512;
Ny = 512;
N = Nx*Ny;
noise_level = 0.07;
nlevel=2;
wv='db8';
dwtmode('per');

% sourse
files=dir('./project/implement/samples');
imds = imageDatastore('./project/implement/samples');
operators = load('-mat','./project/implement/simulated_data/operators/data.mat');
net=load('-mat','./project/net/M3.mat');
validationname={files.name};

% parameters
options.verbose = 1;
options.rel_tol = 1e-5;
options.rel_tol2 = 1e-5;
options.delta = 1;
options.max_iter = 1000;
options.rho = 100;  % 惩罚项
options.delta = 1e-10;% 斜率，step size for the proximal gradient update 需要小于lipschitz continuous gradient

paramtv.max_iter = 100;
paramtv.rel_obj = 1e-3;
paramtv.verbose = 0;
    


% Definition of the measurement operators Phit and Phi
Phi_t = @(x) HS_forward_operator(x,operators.Gw,operators.A); % forward measurement operator
Phi = @(y) HS_adjoint_operator(y,operators.Gw,operators.At,Nx,Ny); %associated adjoint operator

%convergence check
%  uni_matrix = eye(Nx);
%  [U,S,V] = svd(Phi_t(uni_matrix));
%  L = max(max(S*S'));

%calculate eplison simulate 
start_value = 118.3 ; %118.5
step_size = 0.1 ; 
num_steps = 10 ;
eplison_level_matrix = start_value:step_size:((num_steps-1)*step_size+start_value);

result_matrix = zeros(num_steps,size(imds.Files,1));
j=1;% index of simulation_level_matrix
while true
    
    i=1; %figure
   
while hasdata(imds)
    eplison_level = eplison_level_matrix(j);
    % read and compare
    im = read(imds);
    im=im2double(im);
    im=im/max(max(im));

%     figure(1) 
%     imagesc(im),axis image, colorbar, colormap gray;
%     title('Original image.');

    y = Phi_t(im);
    x_project = Phi(y);
    x_project=x_project/max(max(x_project));
%     figure(2) 
%     imagesc(real(x_project)),axis image, colorbar, colormap gray;
%     title('project image.');
    rsnr_before = 20*log10(norm(im(:))/norm(im(:)-x_project(:)));

%% 2. Sparsity operators    
    % sparsity operator 
   

    % Definition of the sparsity operators Psi and Psit
    [alphad,S]=wavedec2(x_project,nlevel,wv);
    Psit = @(x) wavedec2(x, nlevel,wv); 
    Psi = @(x) waverec2(x,S,wv);

    % sparsity test
%         figure(3)
%         a = Psit(im);
%         histogram(a,'Normalization','count');
%         figure(4)
%         histogram(im,'Normalization','count');
%         a_sparseness=sparseness(a);
%         im_sparseness=sparseness(im);

%% 3. noisy operators      
    % add noise operator
    add_noise = @(y) y + (randn(size(y)) + 1i*randn(size(y)))*noise_level/sqrt(2);

    % add noise and define eplison
    y_noise=add_noise(y);
    
    epsilon =eplison_level*norm(y_noise-y);


%% 4. solve the value

    %admm
    [xsol, rsnr, niter] = admm(y_noise,epsilon,Phi_t,Phi,Psi,Psit,options,net,im,paramtv);
    %[xsol, fval, t] = admm_contvdn(y1,epsilon,Phi_t,Phi,options,paramtv)
    %[xsol, rsnr, num] = admm_a(y_noise,epsilon,Phi_t,Phi,Psi,Psit,options,net,im);
    
   % write
   length = size(validationname{i+3});
   name=['./project/implement/reconstruction_images/',validationname{i+3}(1:(length(2)-5)),'.png'];
   imwrite(xsol,name);
    result_matrix(j,i) = rsnr ;
    fprintf("simulate等级=%f,第%d张图片,rsnr值为=%f",eplison_level_matrix(j),i,rsnr);
    i=i+1;
    break;
end 
    j = j+1;
    if(j>size(eplison_level_matrix,2))
       break; 
    end
    reset(imds)
end
 [B,I]  =  sort(mean(result_matrix,2),'descend');
 best_simulation_level = eplison_level_matrix(I(1));
 %save('./project/implement/simulated_data/simulation_level/level_',start_value,'_',(num_steps*step_size+start_value-1),'.mat', 'simulation_level_matrix', 'result_matrix');
        

    
