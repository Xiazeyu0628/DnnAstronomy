clear
clc
ratio = 0.3;  % clean/noisy
sigma = 0.07 ;
add_noise_x = @(x) x + + sigma*randn(size(x));
imds = imageDatastore('./project/data/groundtruth_png/good','ReadFcn',@NormalizeImageResize);
files=dir('./project/data/groundtruth_png/good/*.png');
noisy_name={files.name};

nFiles = length(imds.Files);
RandIndices = randperm(nFiles);
ratio2num = round(ratio*nFiles);

pure_image_indices = RandIndices(1:ratio2num);
pure_image_set = subset(imds,pure_image_indices);
pure_name = {1,size(pure_image_indices,2)};
for m = 1:size(pure_image_indices,2)
pure_name{m} = noisy_name{pure_image_indices(m)};
end
i = 1;
while hasdata(imds)
    x = read(imds);
    x_noisy = add_noise_x(x);
    x_residual = x_noisy - x;
    length = size(noisy_name{i});
    noise_name=['./project/data/noisy_images/',noisy_name{i}(1:(length(2)-4)),'.png'];
    imwrite(x_noisy,noise_name);
    residual_name=['./project/data/residual_images/',noisy_name{i}(1:(length(2)-4)),'.png'];
    imwrite(x_residual,residual_name);
    i = i+1
end

i = 1;
while hasdata(pure_image_set)
    x = read(pure_image_set);
    x_noisy = x;
    x_residual = x_noisy - x;
    length = size(noisy_name{i});
    noise_name=['./project/data/noisy_images/',noisy_name{i}(1:(length(2)-4)),'.png'];
    imwrite(x_noisy,noise_name);
    residual_name=['./project/data/residual_images/',noisy_name{i}(1:(length(2)-4)),'.png'];
    imwrite(x_residual,residual_name);
    i = i+1
end



