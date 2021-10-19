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

% sourse
sample_files=dir('./project/data/samples2/*.png');
%sample_files=dir('./project/data/test_set/*.png');
net_files = dir('./project/net/bayes/*.mat');
net_files_folder = {net_files.folder};
net_files_name = {net_files.name};
imds = imageDatastore('./project/data/samples2');
%imds = imageDatastore('./project/data/test_set');
operators = load('-mat','./project/simulated_data/operators/data512.mat');
validationname={sample_files.name};

% Definition of the measurement operators Phit and Phi
Phi_t = @(x) HS_forward_operator(x,operators.Gw,operators.A); % forward measurement operator
Phi = @(y) HS_adjoint_operator(y,operators.Gw,operators.At,Nx,Ny); %associated adjoint operator

len = size( net_files_name,2);
for k = 1:len
    net = load([net_files_folder{1},'/',net_files_name{k}]);
    net_name_length = size(net_files_name{k},2);
    net_value = str2num(net_files_name{k}(7:10));
    %分析收敛性
%     Mat_x = net.info.ValidationLoss ; 
%     TF = isnan(Mat_x);
%     ind = find(TF==0);
%     plot(Mat_x(ind(3:end)));
%    % plot(net.info.TrainingLoss(100:end));
%     net_convergenve_figure_name = ['./project/result/net/',net_files_name{k}(1:(net_name_length-4)),'.png'];
%     saveas(gcf,net_convergenve_figure_name);
%     close all
    
    %分析去噪音效果
  mkdir('./project/result/reconstruction_images/hybird',net_files_name{k}(1:(net_name_length-4)));
   addpath(['./project/result/reconstruction_images/hybird/',net_files_name{k}(1:(net_name_length-4))]);

    
 %   [denoise_average_rsnr,denoise_result_matrics,denoise_avearge_ssim] = cal_denoise_test(net,net_files_name{k});
    
 %  save(['./project/result/denoise/net',net_files_name{k}(1:(net_name_length-4)),'_',num2str(denoise_average_rsnr),'_',num2str(denoise_avearge_ssim),'.mat'],'denoise_average_rsnr','denoise_result_matrics');
    % parameters 
   options.verbose = 1;
    options.rel_tol = 1e-5;
    options.rel_tol2 = 1e-5;
    options.max_iter = 500;
     
    options.rho =1e-7*0.004/(net_value^2) ; 
    delta_matrix = [1e-7];
    for j = 1:length(delta_matrix)
        options.delta = delta_matrix(j);
        mkdir(['./project/result/reconstruction_images/hybird/',net_files_name{k}(1:(net_name_length-4))],num2str(options.delta));
        addpath(['./project/result/reconstruction_images/hybird/',net_files_name{k}(1:(net_name_length-4)),'/',num2str(options.delta)]);
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
           % options.lambda = 1e-3*norm(alphad,Inf);
           options.lambda = 0.0040;
        %% 3. noisy operators      

            epsilon = sigma*sqrt(M + 2*sqrt(M));
 
            add_noise = @(y) y + (randn(size(y)) + 1i*randn(size(y)))*sigma/sqrt(2);
            y_noise=add_noise(y);
        
           validation_length = size(validationname{i},2);
           
           
           tstart=tic;
           %% 4. solve the value
            % [xsol, rsnr, niter] = admm_prox(y_noise,epsilon,Phi_t,Phi,Psi,Psit,options,net,im,validationname{i+3}(1:(length(2)-5))); 
           %  [xsol, rsnr, niter] = admm_BM3D(y_noise,epsilon,Phi_t,Phi,Psi,Psit,options,im,noise_level);
            %[xsol, rsnr, niter] = admm_hybird(y_noise,epsilon,Phi_t,Phi,Psi,Psit,options,net.trainedNet,im);
        %  [xsol, rsnr, niter,result] = admm_hybird_test(y_noise,epsilon,Phi_t,Phi,Psi,Psit,options,net.trainedNet,im);
            [xsol, rsnr, niter,result] = admm_hybird_convergence(y_noise,epsilon,Phi_t,Phi,Psi,Psit,options,net.trainedNet,im);
        
           
           ssimval = ssim(double(xsol),im);
           reconstruct_time=toc(tstart);
           % write
           image_name=['./project/result/reconstruction_images/hybird/',net_files_name{k}(1:(net_name_length-4)),'/',num2str(options.delta),'/',validationname{i}(1:(validation_length-4)),'_',num2str(rsnr),'_',num2str(ssimval),'.png'];
           imwrite(xsol,image_name);
           
           figure(1)
           scatter(result(1,:),result(2,:));
           point_figure_name = ['./project/result/reconstruction_images/hybird/',net_files_name{k}(1:(net_name_length-4)),'/',num2str(options.delta),'/rsnr_',validationname{i}(1:(validation_length-4)),'.jpg'];
           saveas(1,point_figure_name);
                      
           close all
           result_matrix(1,i) = i ;
           result_matrix(2,i) = rsnr ;
           result_matrix(3,i) = ssimval ;
           result_matrix(4,i) = reconstruct_time ;
           i=i+1;

        end 
        save(['./project/result/reconstruction_images/hybird/',net_files_name{k}(1:(net_name_length-4)),'/',num2str(options.delta),'/',net_files_name{k}(1:(net_name_length-4)),'.mat'],'result_matrix','options');
        reset(imds);
        average_rsnr = zeros(1,size(result_matrix(2,:),2));
        average_ssimval = zeros(1,size(result_matrix(2,:),2));
        average_rsnr(1,1) = mean(result_matrix(2,:));
        average_ssimval(1,1) = mean(result_matrix(3,:));
        Excel_name = ['./project/result/reconstruction_images/hybird/',net_files_name{k}(1:(net_name_length-4)),'/',num2str(options.delta),'/',net_files_name{k}(1:(net_name_length-4)),'.csv'];
        sheet_name = 'sheet1';
        write_content = [result_matrix(1,:);result_matrix(2,:);result_matrix(3,:);result_matrix(4,:);average_rsnr;average_ssimval];
        xlswrite(Excel_name,write_content,sheet_name);
    end
end
        


        

    
