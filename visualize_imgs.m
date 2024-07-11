function [] = visualize_imgs(train_ds,test_ds)
train_labels = train_ds.Labels;
test_labels = test_ds.Labels;

figure;

subplot(2,3,1);
idx = 102;
imshow(train_ds.Files{idx});
title(train_labels(idx))

subplot(2,3,2);
idx = 304;
imshow(train_ds.Files{idx});
title(train_labels(idx))

subplot(2,3,3);
idx = 9;
imshow(train_ds.Files{idx});
title(train_labels(idx))

subplot(2,3,4);
idx = 13;
imshow(test_ds.Files{idx});
title(test_labels(idx))

subplot(2,3,5);
idx = 107;
imshow(test_ds.Files{idx});
title(test_labels(idx))

subplot(2,3,6);
idx = 97;
imshow(test_ds.Files{idx});
title(test_labels(idx))

end