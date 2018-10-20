
clear;
load('../../analysis/train_bow.mat');
labels = labels + 1;
labels_names = {'CA', 'CG', 'CL', 'CN', 'CP', 'CT', 'MF', 'SB', 'SC', 'BG'};

%%

step = floor(size(cluster_patches, 4) / 13);
[x, y] = meshgrid(1:step:size(cluster_patches, 4), 1:step:size(cluster_patches, 4));
x = x(:); y = y(:);

[clust_count, im_count, c, h, w] = size(cluster_patches);
figure;
for c = 1:clust_count
  for i = 1:im_count
    patch = squeeze(mean(squeeze(cluster_patches(c, i, :, :, :))));

    subplot(im_count, clust_count, (i - 1) * clust_count + c);
    ind = train_cluster_patches_locations(c, i);
    if 0
      k = 4;
      lim_patch = patch(max(1,x(ind)-k*step):min(x(ind)+k*step,size(patch,1)), max(1,y(ind)-k*step):min(y(ind)+k*step,size(patch,2)));
      imagesc(lim_patch);
      if i == 1
        title(num2str(c));
      end
    else
      imagesc(patch);
      hold on;
      plot(y(ind), x(ind), 'ro');
    end
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

classes_zoomed = imread('../../analysis/classes_zoomed.png');
r_space = size(classes_zoomed, 1) / 9;
c_space = size(classes_zoomed, 2) / 10;

figure;
for l = 1:9
  l_bows = bows(labels == l, :);

  subplot(2, 2, 1);
  boxplot(l_bows);
  title(labels_names(l));

  for i = 2:4
    subplot(2, 2, i);
    imagesc(classes_zoomed((l - 1) * r_space + 1:l * r_space, i * c_space + 1:(i + 1) * c_space));
  end

  pause;
end

%%

load('../../analysis/test_bow.mat');
labels = labels + 1;

%%

figure;
for l = 1:10
  l_bows = bows(labels == l, :);

  subplot(2, 5, l);
  boxplot(l_bows);
  title(labels_names(l));
end

