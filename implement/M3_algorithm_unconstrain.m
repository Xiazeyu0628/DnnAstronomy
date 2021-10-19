clear 
clc

% constants
Nx = 512;
Ny = 512;
N = Nx*Ny;

isnr = 30 ;%对�??????��??�???�?�?isnr?��?��?�?????sigma�?�?�?�???常�??�???此�??��??
sigma = 0.07; 

nlevel=2;
wv='db8';
dwtmode('per');
M = 30000;

% sourse
files=dir('./project/data/samples2/*.png');

imds = imageDatastore('./project/data/samples2');
%imds = imageDatastore('./project/data/lectures');

operators = load('-mat','./project/simulated_data/operators/data512.mat');
net=load('-mat','./project/net/dncnn-0.07_04_08__10_47.mat');

validationname={files.name};

% parameters �???许�?��?��??????
options.verbose = 1;
options.rel_tol = 1e-5;
options.rel_tol2 = 1e-5;
options.max_iter = 500;

options.rho = 0.001;  % ?�混??�?�?�?没�???��??

%step size for the proximal gradient update ??�?�?�?lipschitz continuous gradient
% admm with denoise with non-differentiable fadelity term size64 :
% val = 1e-7~2e-7
% admm with prox with non-differentiable fadelity term value:val < 1e-7
% admm with denoise with non-differentiable fadelity term   value:val =
% 1e-9~1e-10
options.delta = 3e-7;

% 
% eta should beteen 0-1
options.eta = 0.90 ;
% gamma>1
options.gamma = 1.5 ;

% Definition of the measurement operators Phit and Phi
Phi_t = @(x) HS_forward_operator(x,operators.Gw,operators.A); % forward measurement operator
Phi = @(y) HS_adjoint_operator(y,operators.Gw,operators.At,Nx,Ny); %associated adjoint operator

\

i=1; %figure
result_matrix = zeros(4,size(imds.Files,1));

while hasdata(imds)
    
    % read and compare
    im = read(imds);
    im=im2double(im);
    if im==zeros(512,512)
        im = eps(im);
    else
    im=im/max(max(im));
    end
    
    y = Phi_t(im);
    x_project = Phi(y);
    x_project=x_project/max(max(x_project));
    rsnr_before = 20*log10(norm(im(:))/norm(im(:)-x_project(:)));

%% 2. Sparsity operators    

    % Definition of the sparsity operators Psi and Psit
    [alphad,S]=wavedec2(x_project,nlevel,wv);
    Psit = @(x) wavedec2(x, nlevel,wv); 
    Psi = @(x) waverec2(x,S,wv);
%     V = (rand(1,Nx)<0.5)*2-1;
%     D = diag(V);
%     Psi = @(a) D*waverec2(a,S,wv);
%     Psit = @(x) wavedec2(D\x, nlevel,wv);

       %sparsity test
        a = Psit(im); 
        histogram(abs(a),'Normalization','count');
        [B,I] = sort(a,'descend');
        plot(B);
%         length = size(validationname{i+3});
%         sparsity_name=['./project/result/sparsity/x_project/',validationname{i+3}(1:(length(2)-4)),'.png'];
%         saveas(gcf,sparsity_name);
%         close all
%         figure(4)
%         histogram(im,'Normalization','count');

        %Estimation of the regularization parameter
      % options.lambda = 1e-3*norm(alphad,Inf);
      options.lambda = 16.5;
%% 3. noisy operators      
        sigma = sqrt(options.delta*options.lambda/options.rho);
     %   epsilon = 0;
    %  define eplison and standard diviation sigma
%     sigma = norm(im)/sqrt(Nx)*10^(-isnr/20);
    epsilon = sigma*sqrt(M + 2*sqrt(M));
  
     % add noise operator
    add_noise = @(y) y + (randn(size(y)) + 1i*randn(size(y)))*sigma/sqrt(2);
    y_noise=add_noise(y);
   %% ?��????
   length = size(validationname{i});
   tstart=tic;
   
   %% 4. solve the value
   %  [xsol, rsnr, niter] = admm_prox(y_noise,epsilon,Phi_t,Phi,Psi,Psit,options,net,im,validationname{i}(1:(length(2)-5))); 
   %  [xsol, rsnr, niter] = admm_BM3D(y_noise,epsilon,Phi_t,Phi,Psi,Psit,options,im,noise_level);
   % [xsol, rsnr, niter] = admm_hybird(y_noise,epsilon,Phi_t,Phi,Psi,Psit,options,net.trainedNet,im,validationname{i}(1:(length(2)-4)));
   [xsol, rsnr, niter,result] = admm_hybird_convergence(y_noise,epsilon,Phi_t,Phi,Psi,Psit,options,net.trainedNet,im);
   % [xsol, rsnr, niter] = admm_residual(y_noise,epsilon,Phi_t,Phi,Psi,Psit,options,net.trainedNet,im);
   
   ssimval = ssim(double(xsol),im);
   reconstruct_time=toc(tstart);
   scatter(result(1,:),result(2,:));
   point_figure_name = ['./project/result/reconstruction_images/hybird/unconstraint/',validationname{i}(1:(length(2)-4)),'.jpg'];            saveas(gcf,point_figure_name);
   close all
   
   % write
   name=['./project/result/reconstruction_images/hybird/unconstraint/',validationname{i}(1:(length(2)-4)),'_',num2str(rsnr),'_',num2str(ssimval),'.png'];
  % name=['./project/result/reconstruction_images/prox/',validationname{i}(1:(length(2)-4)),'_',num2str(rsnr),'_',num2str(ssimval),'.png'];
   imwrite(xsol,name);
   
   result_matrix(1,i) = i ;
   result_matrix(2,i) = rsnr ;
   result_matrix(3,i) = ssimval ;
   result_matrix(4,i) = reconstruct_time ;
   i=i+1;   
end 
save(['./project/result/reconstruction_images/hybird/unconstraint/result.mat'],'result_matrix','options');
average_rsnr = zeros(1,size(result_matrix(2,:),2));
average_ssimval = zeros(1,size(result_matrix(2,:),2));
average_rsnr(1,1) = mean(result_matrix(2,:));
average_ssimval(1,1) = mean(result_matrix(3,:));
%Excel_name = ['./project/result/reconstruction_images/hybird/M5_0321.csv'];
Excel_name = ['./project/result/reconstruction_images/hybird/unconstraint/hybird.csv'];
sheet_name = 'sheet1';
write_content = [result_matrix(1,:);result_matrix(2,:);result_matrix(3,:);result_matrix(4,:);average_rsnr;average_ssimval];
xlswrite(Excel_name,write_content,sheet_name);
        
        

        

    
