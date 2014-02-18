function f = rlayne_extract_feature(rgbimg, gabor, schmid)

% Fix image size to 120*45
h = 120;
w = 45;
channels = zeros(h, w, 29);

rgbimg = imresize(rgbimg, [h, w]);

% Different color channels
hsvimg = rgb2hsv(rgbimg);
ycbcrimg = rgb2ycbcr(rgbimg);

channels(:, :, 1:3) = rgbimg;
channels(:, :, 4:5) = hsvimg(:, :, 1:2);
channels(:, :, 6:8) = ycbcrimg;

% Texture filter on v-channel
lum  = hsvimg(:,:,3);

for i = 1:length(gabor)
    channels(:, :, 8+i) = abs(imfilter(lum, gabor{i}, 'symmetric'));
end
for i = 1:length(schmid)
    channels(:, :, 16+i) = abs(imfilter(lum, schmid{i}, 'symmetric'));
end

% Histogram distribution
n_bins = 16;
strip_height = 20;
n_strips = h / strip_height;

f = zeros(size(channels, 3), n_strips, n_bins);

parfor i = 1:size(channels, 3)
    for j = 1:n_strips
        l = (j-1) * strip_height + 1;
        r = j * strip_height;
        strip = channels(l:r, :, i);
        f(i, j, :) = hist(strip(:), n_bins);
    end
end

f = f(:)';

end