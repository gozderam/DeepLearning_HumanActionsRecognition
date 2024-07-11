function [features] = part1_hog_fe(ds, cell_size)
%APP1_FEATURE_EXTRACTION Summary of this function goes here
%  ds - data store

img = part1_img_preprocessing(readimage(ds, 1));
[hog_4x4, ~] = extractHOGFeatures(img,'CellSize',cell_size);
hog_feature_size = length(hog_4x4);

numImages = numel(ds.Files);
features = zeros(numImages,hog_feature_size,'single');

for i = 1:numImages
    img = readimage(ds,i);
    img = part1_img_preprocessing(img);
    features(i, :) = extractHOGFeatures(img,'CellSize',cell_size);  
end

end