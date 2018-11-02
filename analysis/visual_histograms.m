
clear;
load('../results/train_bow.mat');
labels = labels + 1;
labels_names = {'CA', 'CG', 'CL', 'CN', 'CP', 'CT', 'MF', 'SB', 'SC', 'BG'};

%%

step = floor(size(cluster_patches, 4) / 5) - 1;
[x, y] = meshgrid(1:step:size(cluster_patches, 4), 1:step:size(cluster_patches, 4));
x = x(:); y = y(:);

[clust_count, im_count, c, h, w] = size(cluster_patches);
figure;
for c = 1:clust_count
  for i = 1:im_count
    patch = squeeze(mean(squeeze(cluster_patches(c, i, :, :, :))));

    subplot(im_count, clust_count, (i - 1) * clust_count + c);
    imshow(patch);
    hold on;
    ind = train_cluster_patches_locations(c, i);
    plot(y(ind), x(ind), 'ro');
  end
end

%%

figure;
for l = 1:10
  l_bows = bows(labels == l, :);

  subplot(2, 5, l);
  boxplot(l_bows);
  title(labels_names(l));
end

%%

load('results/test_bow.mat');
labels = labels + 1;

%%

figure;
for l = 1:10
  l_bows = bows(labels == l, :);

  subplot(2, 5, l);
  boxplot(l_bows);
  title(labels_names(l));
end

