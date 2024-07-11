function [x_out] = part1_img_preprocessing(x)
% APP_IMG_TRANSFORM 
%   x - input image
%   x_out - image after data preprocessing

% resize parameters 
targetSize = [256,256];

% gaussian smoothing parameters
smooth_gamma = 0.5;

x_out = imresize(x, targetSize);
x_out = imgaussfilt(x_out, smooth_gamma);
x_out = histeq(x_out);
x_out = im2gray(x_out);
x_out = imbinarize(x_out);
end