clear 
clc
addpath('utils/');
addpath('utils/lib/');
run('utils/lib/irt/setup.m');
% constants
Nx = 512;
Ny = 512;
N = Nx*Ny;

isnr = 30 ;
sigma = 0.07; 

nlevel=2;
wv='db8';
dwtmode('per');
M = 30000;

net=load('-mat','./project/net/dncnn-0.07_04_10__03_34.mat');
% sourse
sample_files=dir('./project/data/samples2/*.png');

imds = imageDatastore('./project/data/samples2');

operators = load('-mat','./project/simulated_data/operators/data512.mat');

validationname={sample_files.name};

% Definition of the measurement operators Phit and Phi
Phi_t = @(x) HS_forward_operator(x,operators.Gw,operators.A); % forward measurement operator
Phi = @(y) HS_adjoint_operator(y,operators.Gw,operators.At,Nx,Ny); %associated adjoint operator


   
%parameters 
options.verbose = 1;
options.rel_tol = 1e-5;
options.rel_tol2 = 1e-5;
options.max_iter = 500;
options.delta = 1e-7;

rho_matrix = [1000];
for j = 1:length(rho_matrix)
    options.rho = rho_matrix(j);
    mkdir(['./project/result/reconstruction_images/prox/',num2str(options.rho)]);
    addpath(['./project/result/reconstruction_images/prox/',num2str(options.rho)]);
    
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
       
        options.lambda = 1e-3*norm(alphad,Inf);
    %% 3. noisy operators      

        epsilon = sigma*sqrt(M + 2*sqrt(M));

        add_noise = @(y) y + (randn(size(y)) + 1i*randn(size(y)))*sigma/sqrt(2);
        y_noise=add_noise(y);

        validation_length = size(validationname{i},2);


       tstart=tic;
       %% 4. solve the value
        % [xsol, rsnr, niter,result] = admm_prox(y_noise,epsilon,Phi_t,Phi,Psi,Psit,options,im); 
       %  [xsol, rsnr, niter] = admm_BM3D(y_noise,epsilon,Phi_t,Phi,Psi,Psit,options,im,noise_level);
        %[xsol, rsnr, niter] = admm_hybird(y_noise,epsilon,Phi_t,Phi,Psi,Psit,options,net.trainedNet,im);
       % [xsol, rsnr, niter,result,noise_matrix] = admm_hybird_test(y_noise,epsilon,Phi_t,Phi,Psi,Psit,options,net.trainedNet,im);
        [xsol, rsnr, niter,result] = admm_hybird_convergence(y_noise,epsilon,Phi_t,Phi,Psi,Psit,options,net.trainedNet,im);
     
       ssimval = ssim(double(xsol),im);
       reconstruct_time=toc(tstart);
       
       % write
       image_name=['./project/result/reconstruction_images/prox/',num2str(options.rho),'/',validationname{i}(1:(validation_length-4)),'_',num2str(rsnr),'_',num2str(ssimval),'.png'];
       imwrite(xsol,image_name);

       figure(1)
       scatter(result(1,:),result(2,:));
       point_figure_name = ['./project/result/reconstruction_images/prox/',num2str(options.rho),'/rsnr_',validationname{i}(1:(validation_length-4)),'.jpg'];
       saveas(1,point_figure_name);

       close all
       result_matrix(1,i) = i ;
       result_matrix(2,i) = rsnr ;
       result_matrix(3,i) = ssimval ;
       result_matrix(4,i) = reconstruct_time ;
       i=i+1;

    end 
    save(['./project/result/reconstruction_images/prox/',num2str(options.rho),'/prox.mat'],'result_matrix','options');
    reset(imds);
    average_rsnr = zeros(1,size(result_matrix(2,:),2));
    average_ssimval = zeros(1,size(result_matrix(2,:),2));
    average_rsnr(1,1) = mean(result_matrix(2,:));
    average_ssimval(1,1) = mean(result_matrix(3,:));
    Excel_name = ['./project/result/reconstruction_images/prox/',num2str(options.rho),'/prox.csv'];
    sheet_name = 'sheet1';
    write_content = [result_matrix(1,:);result_matrix(2,:);result_matrix(3,:);result_matrix(4,:);average_rsnr;average_ssimval];
    xlswrite(Excel_name,write_content,sheet_name);
end




        

    
