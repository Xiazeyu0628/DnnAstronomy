%% Operators for monochromatic Radio-Interferometry
% Setup and initialization
clc;
clear all;
close all;
addpath('utils/');
addpath('utils/lib/');

run('utils/lib/irt/setup.m');


%% Definition of the operators
% The measurement operators can be modelled as 
%               Phi_t = Gw*A
% where A and G are matrices that will depend on a number of parameters,
% such as the frequency of observation f. 
% Given the shape of the image, one can build the adjoint operator At. 
% In the case of the super resolution,
% the acquisition process is slightly modified and we need to precise that
% when defining the operators with the "super_resolution" parameter set to
% True (1).
%[A, At, Gw] = generate_data_basic(Nx,Ny,f,1);

%Phi_t = @(x) HS_forward_operator(x,Gw,A);           % Forward (measurement) operator
%Phi = @(y) HS_adjoint_operator(y,Gw,At,Nx,Ny);      % Adjoint operator


%% Example

Nx = 512;                                    % get the dimensions
Ny = 512;
f = 1.4;                                            % frequency
super_res=1;                                        % super resolution: to be set to false (0)


% [A, At, Gw] = generate_data_basic(Nx,Ny,f,super_res,0);
% save('./project/simulated_data/operators/data512.mat', 'Gw', 'At', 'A');
operators = load('-mat','./project/simulated_data/operators/data512.mat');
Gw = operators.Gw;
A = operators.A;
At = operators.At;
Phi_t = @(x) HS_forward_operator(x,Gw,A);
Phi = @(y) HS_adjoint_operator(y,Gw,At,Nx,Ny);


sigma = 0.07;
% add_noise_y = @(y) y + (randn(size(y))+1i*randn(size(y)))*sigma/sqrt(2);






% 
files=dir('./project/data/groundtruth_fit');
imagesname={files.name};

%training-set origin x0
fit_dataset = imageDatastore('./project/data/groundtruth_fit');

i = 1 ;
while hasdata(fit_dataset)

x0 = read(fit_dataset); % read an image
row_num = size(x0,1); %hang
col_num = size(x0,2);
x0_incese = zeros(512,512);

% Cut size to 512*512
if((row_num~=512)|(col_num~=512))
    row_start = round((row_num-512)/2);
    col_start = round((col_num-512)/2);
    for k = 1:512
        for j = 1:512
        x0_incese(k,j) = x0(k+row_start,j+col_start);
        end
    end
    x0 = x0_incese;
end


%judge whether pure black part is too big
patchsize = 64 ;
flag = 0;
for k = 1:Nx-patchsize % hang 
    for l = 1:Ny-patchsize 
        if x0(k:1:k+patchsize-1,l)==zeros(patchsize,1) 
            if x0(k:1:k+patchsize-1,l:1:l+patchsize-1)==zeros(patchsize,patchsize) %????????63????0
                flag = 1;  
                break;
            end
        end
    end
    if flag == 1 
       break;
    end
end
    
% ?????
if (flag == 1)
    length = size(imagesname{i+2});
    groundtruth_name = ['./project/data/groundtruth_png/bad/',imagesname{i+2}(1:(length(2)-5)),'.png'];
    imwrite(x0,groundtruth_name);
    i=i+1
else
    length = size(imagesname{i+2});
    groundtruth_name = ['./project/data/groundtruth_png/good/',imagesname{i+2}(1:(length(2)-5)),'.png'];
    imwrite(x0,groundtruth_name);
    y = Phi_t(x0);
    xbp = real(Phi(y));  
    xbp_name = ['./project/data/project_images/',imagesname{i+2}(1:(length(2)-5)),'.png'];
    xbp = xbp/max(xbp(:)); %normalization
    imwrite(xbp,xbp_name);
    
    % x 的噪声图片，用于训练残差去噪网络
%     y = add_noise_y(y);
%     xbp_noise = real(Phi(y));    %project
%     xbp_noise_name = ['./project/data/noisy_images/',imagesname{i+2}(1:(length(2)-5)),'.png'];
%     xbp_noise = xbp_noise/max(xbp_noise(:)); %normalization
%     imwrite(xbp_noise,xbp_noise_name);
%     rsnr
    
    i=i+1
end
end