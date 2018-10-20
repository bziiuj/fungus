
clear;
types = {'CA', 'CG', 'CL', 'CN', 'CP', 'CT', 'MF', 'SB', 'SC', 'BG'};
png_folder = '../../data/pngs';
mask_folder = '../../data/masks';
mkdir('../../analysis');

%%

% difference between classes (mean for all 3 channel)
k = 3;
aim = uint8(zeros(k * 9 *  360, k * 10 * 576));
for ti = 1:9
  for ii = 1:10
    filename = [png_folder, '/', types{ti}, '/', types{ti}, num2str(2 * ii), '.png'];
    if exist(filename, 'file')
      pim = imread(filename);
      pim = uint16(mean(pim, 3));
      pim = imresize(pim, [k * 360, k * 576]);
      aim((ti - 1) * k * 360 + 1:ti * k * 360, (ii - 1) * k * 576 + 1:ii * k * 576) = pim;
    end
  end
end
% imshow(aim);
imwrite(aim, '../../analysis/classes.png');

%%

% difference between classes (mean for all 3 channel) zoomed
k = 3;
aim = uint8(zeros(k * 9 *  360, k * 10 * 576));
for ti = 1:9
  for ii = 1:10
    filename = [png_folder, '/', types{ti}, '/', types{ti}, num2str(2 * ii), '.png'];
    if exist(filename, 'file')
      pim = imread(filename);
      if exist([mask_folder, '/', types{ti}, '/', types{ti}, num2str(2 * ii), '.png'], 'file')
        mask = imread([mask_folder, '/', types{ti}, '/', types{ti}, num2str(2 * ii), '.png']);
        [r, c] = find(mask == 2);
        r = r(floor(end / 2)); c = c(floor(end / 2));
        br = max(r - 2 * 360 / 2 + 1, 1);
        er = min(r + 2 * 360 / 2, size(pim, 1));
        bc = max(c - 2 * 576 / 2 + 1, 1);
        ec = min(c + 2 * 576 / 2, size(pim, 2));
        randomPart = pim(br:er, bc:ec);
        randomPart = imresize(randomPart, [k * 360, k * 576]);
        aim((ti - 1) * k * 360 + 1:ti * k * 360, (ii - 1) * k * 576 + 1:ii * k * 576) = randomPart;
      end
    end
  end
end
% imshow(aim);
imwrite(aim, '../../analysis/classes_zoomed.png');

%%

% difference between classes (mean for all 3 channel) zoomed (different
% layout)
figure;
for ii = 1:10
  for ti = 1:9
    filename = [png_folder, '/', types{ti}, '/', types{ti}, num2str(2 * ii), '.png'];
    if exist(filename, 'file')
      pim = imread(filename);
      mask = imread([mask_folder, '/', types{ti}, '/', types{ti}, num2str(2 * ii), '.png']);
      [r, c] = find(mask == 2);
      r = r(floor(end / 2)); c = c(floor(end / 2));
      br = max(r - 360 + 1, 1);
      er = min(r + 360, size(pim, 1));
      bc = max(c - 576 + 1, 1);
      ec = min(c + 576, size(pim, 2));
      randomPart = pim(br:er, bc:ec);

      subplot(2, 5, ti);
      imshow(randomPart);
      title(types{ti});
    end
  end
  pause;
end
