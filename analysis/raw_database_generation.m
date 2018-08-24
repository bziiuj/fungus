
types = {'CA', 'CG', 'CL', 'CN', 'CP', 'CT', 'GN', 'MF', 'SB', 'SC'};

%%

% differences between channels
if 0
  k = 3;
  aim = uint8(zeros(k * 10 *  360, k * 3 * 576));
  for ti = 1:10
    for ii = 1
      im = imread([types{ti}, '/', types{ti}, num2str(ii), '.tif']);
      lim = stretchlim(uint16(mean(im, 3)), [0.01, 0.99]);
      for ci = 1:3
        pim = im(:, :, ci);
        pim = imadjust(pim, lim);
        pim = uint8(pim / 256);
        pim = imresize(pim, [k * 360, k * 576]);
        if ii == 1
          aim((ti - 1) * k * 360 + 1:ti * k * 360, (ci - 1) * k * 576 + 1:ci * k * 576) = pim;
        elseif ii == 6
          aim((ti - 1) * k * 360 + 1:ti * k * 360, (ci - 1 + 3) * k * 576 + 1:(ci + 3) * k * 576) = pim;
        elseif ii == 11
          aim((ti - 1) * k * 360 + 1:ti * k * 360, (ci - 1 + 6) * k * 576 + 1:(ci + 6) * k * 576) = pim;
        end
      end
    end
  end
  % imshow(aim);
  imwrite(aim, 'channels.png');
end

%%

% difference between classes (mean for all 3 channel)
if 0
  k = 3;
  aim = uint8(zeros(k * 10 *  360, k * 10 * 576));
  for ti = 1:10
    for ii = 1:10
      pim = imread([types{ti}, '/', types{ti}, num2str(2 * ii), '.tif']);
      pim = uint16(mean(pim, 3));
      lim = stretchlim(pim, [0.01, 0.99]);
      pim = imadjust(pim, lim);
      pim = uint8(pim / 256);
      pim = imresize(pim, [k * 360, k * 576]);
      
      aim((ti - 1) * k * 360 + 1:ti * k * 360, (ii - 1) * k * 576 + 1:ii * k * 576) = pim;
    end
  end
  % imshow(aim);
  imwrite(aim, 'classes.png');
end

%%

% difference between classes (mean for all 3 channel) with hist norm
if 0
  k = 3;
  aim = uint8(zeros(k * 10 *  360, k * 10 * 576));
  for ti = 1:10
    for ii = 1:10
      pim = imread(['pngs/', types{ti}, '/', types{ti}, 2 * ii, '.png']);
      pim = imresize(pim, [k * 360, k * 576]);
      
      aim((ti - 1) * k * 360 + 1:ti * k * 360, (ii - 1) * k * 576 + 1:ii * k * 576) = pim;
    end
  end
  % imshow(aim);
  imwrite(aim, 'classes_hist.png');
end

% difference between classes (mean for all 3 channel) with hist norm,
% zoomed
if 0
  k = 3;
  aim = uint8(zeros(k * 10 *  360, k * 10 * 576));
  for ti = 1:10
    for ii = 1:10
      pim = imread(['pngs/', types{ti}, '/', types{ti}, num2str(2 * ii), '.png']);
      mask = imread(['masks/', types{ti}, '/', types{ti}, num2str(2 * ii), '.png']);
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
  % imshow(aim);
  imwrite(aim, 'classes_hist_zoomed.png');
end

%%

% save pngs with better contrast
if 0
  for ti = 1:10
    mkdir('pngs/', types{ti});
    paths = dir([types{ti}, '/', '*.tif']);
    for ii = 1:length(paths)
      iiStr = paths(ii).name(3:end-4);
      pim = imread([types{ti}, '/', types{ti}, iiStr, '.tif']);
      pimMean = uint16(mean(pim, 3));
      lim = stretchlim(pimMean, [0.01, 0.99]);
      pimMean = imadjust(pimMean, lim);
      pimMean = histeq(pimMean,  0:65535);
      pimMean = uint8(pimMean / 256);

      imwrite(pimMean, ['pngs/', types{ti}, '/', types{ti}, iiStr, '.png'])
    end
  end
end

% save masks
if 1
  k = 3;
  aim = uint8(zeros(k * 10 *  360, k * 10 * 576));
  for ti = 1:10
    mkdir('new_masks/', types{ti});
    paths = dir(['raw/', types{ti}, '/', '*.tif']);
    for ii = 1:length(paths)
      iiStr = paths(ii).name(3:end-4);
      pim = imread(['raw/', types{ti}, '/', types{ti}, iiStr, '.tif']);
      pim = uint16(mean(pim, 3));
      lim = stretchlim(pim, [0.01, 0.99]);
      pim = imadjust(pim, lim);
      pim = uint8(pim / 256);
      pimBin = pim < 128;

      s = 250;

      intPim = integralImage(pimBin);
      avgH = integralKernel([1 1 2*s+1 2*s+1], 1/1001^2);
      avgPim = integralFilter(intPim, avgH);
      bgdPimBin = avgPim < 0.01;
      fgdPimBin = avgPim > max(avgPim(:)) / 2;

      bgdFgdUnk = uint8(zeros(size(pim)));
      bgdFgdUnk(s+1:end-s, s+1:end-s) = bgdPimBin + 2 * max(fgdPimBin - bgdPimBin, 0);

      overPim = double(pim / 256);
      overPim(s+1:end-s, s+1:end-s) = double(overPim(s+1:end-s, s+1:end-s)) .* (1 - fgdPimBin * 0.5);
      overPim(s+1:end-s, s+1:end-s) = double(overPim(s+1:end-s, s+1:end-s)) .* (1 - bgdPimBin * 0.25);

      imwrite(bgdFgdUnk, ['new_masks/', types{ti}, '/', types{ti}, iiStr, '.png'])

      % subplot(1, 2, 1);
      % imshow(overPim);
      % subplot(1, 2, 2);
      % imagesc(bgdFgdUnk);
    end
  end
end
