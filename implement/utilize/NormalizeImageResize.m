function img_res = NormalizeImageResize(file)
%   This function takes as input:
%       - file: an image directory
%   It returns:
%       - img_res: the image in range [0,1] with dimensions [256x256].

   img = imread(file);
   img_res = im2double(img);
    
end
