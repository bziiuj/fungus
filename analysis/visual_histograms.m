close all;
clear;
clc;

main_dir = '/Users/bartek/Desktop/Fungus/analysis';

load([main_dir, '/train_bow.mat']);
labels = labels + 1;
labels_names = {'CA', 'CG', 'CL', 'CN', 'CP', 'CT', 'MF', 'SB', 'SC', 'BG'};

%%

step = floor(size(cluster_patches, 4) / 13) - 1;
[row_loc, col_loc] = meshgrid(1:step:size(cluster_patches, 4), 1:step:size(cluster_patches, 4));
row_loc = row_loc(:); col_loc = col_loc(:);

[clust_count, im_count, c, h, w] = size(cluster_patches);
im_count = 25;
s = 100;

for c = 1:clust_count
  sub_patches = uint8(zeros(2*s, 2*s, 3, clust_count));

  fig = figure;
  for i = 1:im_count
    patch = cluster_patches(c, i, :, :, :);
    patch = squeeze(patch);

    ind = train_cluster_patches_locations(c, i);
    %imshow(patch);
    %hold on;
    %plot(col_loc(ind), row_loc(ind), 'ro');

    % crop part with location
    if row_loc(ind) - s < 1
      row_b = 1; row_e = 2 * s;
    elseif row_loc(ind) + s > size(patch, 1)
      row_b = size(patch, 1) - 2 * s + 1; row_e = size(patch, 1);
    else
      row_b = row_loc(ind) - s + 1; row_e = row_loc(ind) + s;
    end
    if col_loc(ind) - s < 1
      col_b = 1; col_e = 2 * s;
    elseif col_loc(ind) + s > size(patch, 2)
      col_b = size(patch, 1) - 2 * s + 1; col_e = size(patch, 2);
    else
      col_b = col_loc(ind) - s + 1; col_e = col_loc(ind) + s;
    end
    sub_patch = patch(row_b:row_e, col_b:col_e, :);

    % subplot(1, 2, 1); imshow(patch); hold on; plot(col_loc(ind), row_loc(ind), 'ro');
    % subplot(1, 2, 2); imshow(sub_patch);

    sub_patches(:, :, :, i) = sub_patch;
  end

  sub_patches_montage = montage(sub_patches, 'Size', [5, 5]);
  title(num2str(c));
  saveas(fig, [main_dir, num2str(c), '.png']);
  pause;
  close all;
end

%%

figure;
for l = 1:10
  l_bows = bows(labels == l, :);

  subplot(2, 5, l);
  boxplot(l_bows);
  title(labels_names(l));
end
