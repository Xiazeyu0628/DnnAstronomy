function img_residual = NormalizeImageResize_residual(file)
%   This function takes as input:
%       - file: an image directory
%   It returns:
%       - img_res: the image in range [0,1] with dimensions [256x256].
   sigma = 0.07;
   img = imread(file);
   img_res = im2double(img);
   img_noise = img_res + sigma*randn(size(img_res));
   img_residual = img_noise - img_res;
   
    
end