function [features] = part1_lbp_fe(ds, cell_size)
%APP1_FEATURE_EXTRACTION Summary of this function goes here
%  ds - data store

img = part1_img_preprocessing(readimage(ds, 1));
lbp_4x4 = extractLBPFeatures(img,'CellSize',cell_size);
lbp_feature_size = length(lbp_4x4);

numImages = numel(ds.Files);
features = zeros(numImages,lbp_feature_size,'single');

for i = 1:numImages
    img = readimage(ds,i);
    img = part1_img_preprocessing(img);
    features(i, :) = extractLBPFeatures(img,'CellSize',cell_size);  
end

end