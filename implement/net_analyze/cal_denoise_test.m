function [avearge_rsnr,result_matrix,avearge_ssim] = cal_denoise_test(net,net_files_name)
net_name_length = size(net_files_name,2);
mkdir('./project/data/awgn_denoised_images/',net_files_name(1:(net_name_length-4)));
files=dir('./project/data/denoise_samples/*.png');
validationname={files.name};
imds = imageDatastore('./project/data/denoise_samples','ReadFcn',@NormalizeImageResize);
imds_noisy = imageDatastore('./project/data/denoise_samples','ReadFcn',@NormalizeImageResize_addnoise);
i = 1;
result_matrix = zeros(3,size(imds.Files,1));
while hasdata(imds)
%% 计算部分
    im = read(imds);
    x_noise = read(imds_noisy);  
    xsol = cell2mat(compute_net(net.trainedNet,x_noise,32));
    xsol = (xsol/max(max(xsol)));
%     x_residual = cell2mat(compute_net(net.trainedNet,x_noise,64));
%     xsol = x_noise - x_residual;
    
    rsnr = 20*log10(norm(im(:))/norm(im(:)-xsol(:)));
    ssimval = ssim(double(xsol),im);
    result_matrix(1,i) = i;
    result_matrix(2,i) = rsnr;
    result_matrix(3,i) = ssimval;
    length = size(validationname{i});
    
    clean_name=['./project/data/awgn_denoised_images/',net_files_name(1:(net_name_length-4)),'/',validationname{i}(1:(length(2)-4)),'_',num2str(rsnr),'_',num2str(ssimval),'.png'];
    imwrite(xsol,clean_name);
    i = i+1
end
    avearge_rsnr = mean(result_matrix(2,:));
    avearge_ssim = mean(result_matrix(3,:));
end

