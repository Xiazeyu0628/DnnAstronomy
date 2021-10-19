function img_noise = NormalizeImageResize_addnoise(file)
%   This function takes as input:
%       - file: an image directory
%   It returns:
%       - img_res: the image in range [0,1] with dimensions [256x256].
   sigma = 0.09;
   img = imread(file);
   img_res = im2double(img);
   noise = sigma*randn(size(img_res));
   noise_level = norm(noise)/norm(randn(size(img_res)));
   img_noise = img_res + noise;
    
end