function [output] = compute_BMED(I,n,noise_level)

% This function takes as input:
%   - net: a neural network
%   - I: an image
%   - n: a patchsize
% It returns:
%   - o: a picture built by the application of the network on patches of n
%   by n of the full image I.
imSz = size(I);
patchSz = [n n];

if mod(imSz(2),n)==0
    xIdxs = [1:patchSz(2):(imSz(2)+1)];
    yIdxs = [1:patchSz(1):(imSz(1)+1)];
    patches = cell(length(yIdxs)-1,length(xIdxs)-1);
    for i = 1:length(yIdxs)-1
        Isub = I(yIdxs(i):yIdxs(i+1)-1,:);
        for j = 1:length(xIdxs)-1
            sub_picture = Isub(:,xIdxs(j):xIdxs(j+1)-1);
            % Here compute the output of the network on each patch
            if size(sub_picture) == patchSz
                patches{i,j} = BM3D(sub_picture, noise_level);
            end
        end
    end
    output = patches;
else
    xIdxs = [1:patchSz(2):(imSz(2)+1)];
    yIdxs = [1:patchSz(1):(imSz(1)+1)];

end
end

