clear
clc
data_files = dir('./project/result/data/net_data/1e-7/*.mat');
data_files_folder = {data_files.folder};
data_files_name =  {data_files.name};

sample_files=dir('./project/data/samples2/*.png');
validationname={sample_files.name};

imds = imageDatastore('./project/data/samples2');

%���ݴ������
noise_value_matrix = zeros(1,size(data_files,1));
rsnr_value_matrix = zeros(size(data_files,1),size(sample_files,1)); % n*24
ssim_value_matrix = zeros(size(data_files,1),size(sample_files,1));
time_value_matrix = zeros(size(data_files,1),size(sample_files,1));
mean_rsnr_value = zeros(size(data_files,1),1);
mean_ssim_value = zeros(size(data_files,1),1);
mean_time_value = zeros(size(data_files,1),1);

%���ݶ���
for i = 1:size(data_files,1)
    data = load([data_files_folder{i},'/',data_files_name{i}]) ;
    noise_value = str2double(data_files_name{i}(7:10));
    noise_value_matrix(1,i) = noise_value; 
    rsnr_value_matrix(i,:) = data.result_matrix(2,:);
    ssim_value_matrix(i,:) = data.result_matrix(3,:);
    time_value_matrix(i,:) = data.result_matrix(4,:);
    mean_rsnr_value(i,1) = mean(rsnr_value_matrix(i,:));
    mean_ssim_value(i,1) = mean(ssim_value_matrix(i,:));
    mean_time_value(i,1) = mean(time_value_matrix(i,:));
end

%ͼƬ����

for j = 1:size(sample_files,1)
pic_name = validationname{j};
rsnr_value = rsnr_value_matrix(:,j);
ssim_value = ssim_value_matrix(:,j);
time_value = time_value_matrix(:,j);

figure(1)
subplot(2,1,1)
title([pic_name,'_rsnr']);
plot(noise_value_matrix,rsnr_value');
subplot(2,1,2)
title([pic_name,'_ssim']);
plot(noise_value_matrix,ssim_value');
saveas(1,['./project/result/data/rsnr_ssim/',pic_name]);
figure(2)
plot(noise_value_matrix,time_value');
saveas(2,['./project/result/data/time/',pic_name]);
j = j+1;
close all
end

% figure(1)
% subplot(2,1,1)

plot(noise_value_matrix,mean_rsnr_value');
title('average_rsnr vs denoising level');
xlabel('denoising level')
ylabel('rsnr')

% subplot(2,1,2)
% title('average_ssim');
plot(noise_value_matrix,mean_ssim_value');
title('average ssim vs denoising level');
xlabel('denoising level')
ylabel('ssim')

figure(2)
plot(noise_value_matrix,mean_time_value');
saveas(1,['./project/result/data/rsnr_ssim/average_rsnr_ssim.png']);
saveas(2,['./project/result/data/time/average_time.png']);
close all
