close all
clear
data_files_step1 = dir('./project/result/data/net_data/1e-7/*.mat');
data_files_folder_step1 = {data_files_step1.folder};

data_files_name =  {data_files_step1.name};

data_files_step2 = dir('./project/result/data/net_data/2e-7/*.mat');
data_files_folder_step2 = {data_files_step2.folder};


sample_files=dir('./project/data/samples2/*.png');

rsnr_value_matrix = zeros(2,24); % 2*24
ssim_value_matrix = zeros(2,24);
mean_rsnr_value_matrix = zeros(2,5);
ssim_rsnr_value_matrix = zeros(2,5);

for i = 1:size(data_files_step1,1)
data_1 = load([data_files_folder_step1{i},'/',data_files_name{i}]) ;
data_2 = load([data_files_folder_step2{i},'/',data_files_name{i}]) ;
rsnr_value_matrix(1,:) = data_1.result_matrix(2,:);
rsnr_value_matrix(2,:) = data_2.result_matrix(2,:);
ssim_value_matrix(1,:) = data_1.result_matrix(3,:);
ssim_value_matrix(2,:) = data_2.result_matrix(3,:);

mean_rsnr_value_matrix(1,i) = mean(rsnr_value_matrix(1,:));
mean_rsnr_value_matrix(2,i) = mean(rsnr_value_matrix(2,:));

mean_ssim_value_matrix(1,i) = mean(ssim_value_matrix(1,:));
mean_ssim_value_matrix(2,i) = mean(ssim_value_matrix(2,:));

figure(1)
hold on
title(['rsnr value for denosing network with sigma =',data_files_name{i}(7:10)]);
plot(1:1:24,rsnr_value_matrix(1,:));
plot(1:1:24,rsnr_value_matrix(2,:));
legend('1e-7','2e-7');

figure(2)
hold on 
title(['ssim value for denosing network with sigma =',data_files_name{i}(7:10)]);
plot(1:1:24,ssim_value_matrix(1,:));
plot(1:1:24,ssim_value_matrix(2,:));
legend('1e-7','2e-7');

saveas(1,['./project/result/data/trend/rsnr_for_simga = ',data_files_name{i}(7:10),'.png']);
saveas(2,['./project/result/data/trend/ssim_for_simga = ',data_files_name{i}(7:10),'.png']);
close all
end

figure(1)
hold on 
title(['average rsnr value for denosing network with sigma =',data_files_name{i}(7:10)]);
plot(1:1:5,mean_rsnr_value_matrix(1,:));
plot(1:1:5,mean_rsnr_value_matrix(2,:));
legend('1e-7','2e-7');

figure(2)
hold on 
title(['average ssim value for denosing network with sigma =',data_files_name{i}(7:10)]);
plot(mean_ssim_value_matrix(1,:));
plot(mean_ssim_value_matrix(2,:));
legend('1e-7','2e-7');

saveas(1,['./project/result/data/trend/average_rsnr_for_simga = ',data_files_name{i}(7:10),'.png']);
saveas(2,['./project/result/data/trend/average_ssim_for_simga = ',data_files_name{i}(7:10),'.png']);
close all
