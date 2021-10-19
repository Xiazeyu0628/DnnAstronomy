clear
clc
files=dir('./project/data/denoise_samples2/*.png');
validationname={files.name};
imds = imageDatastore('./project/data/denoise_samples2','ReadFcn',@NormalizeImageResize);
imds_noisy = imageDatastore('./project/data/denoise_samples2','ReadFcn',@NormalizeImageResize_addnoise);
%net_value = num2str(-15.4363);
%net_name = ['./project/net/',net_value,'.mat'];
net_name =  ['./project/net/dncnn-0.07_04_04__03_58.mat'];
net=load(net_name);
i = 1;
result_matrix = zeros(3,size(imds.Files,1));
while hasdata(imds)
%% 计算部分
    im = read(imds);
    x_noise = read(imds_noisy);  
%     xsol = cell2mat(compute_net(net.trainedNet,x_noise,64));
%     xsol = (xsol/max(max(xsol)));
     x_residual = cell2mat(compute_net(net.trainedNet,x_noise,256));
    xsol = x_noise - x_residual;
    rsnr = 20*log10(norm(im(:))/norm(im(:)-xsol(:)));
    ssimval = ssim(double(xsol),im);
    result_matrix(1,i) = i;
    result_matrix(2,i) = rsnr;
    result_matrix(3,i) = ssimval;
%% 图片写入
    length = size(validationname{i});
    clean_name=['./project/data/awgn_denoised_images/dncnn-0.07_04_04__03_58/',validationname{i}(1:(length(2)-4)),'_',num2str(rsnr),'_',num2str(ssimval),'.png'];
    imwrite(xsol,clean_name);
  residual_name = ['./project/data/awgn_denoised_residual_images/',validationname{i}(1:(length(2)-4)),'.png'];
  imwrite(x_residual,residual_name);
    i = i+1
end
    avearge_rsnr = mean(result_matrix(2,:));
    avearge_ssim = mean(result_matrix(3,:));
  %  save_name = ['./project/result/denoise/',net_value,'_',num2str(avearge_rsnr),'.mat'];
  save_name = ['./project/result/denoise/dncnn-0.07_04_04__03_58','_',num2str(avearge_rsnr),'_',num2str(avearge_ssim),'.mat'];
    save(save_name,'result_matrix','avearge_rsnr');