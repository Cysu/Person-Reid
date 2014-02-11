function masked_img = mask_bodyparts(img, parts, model_coarse, model_fine)

labelmap = decompositionalLayer(img, model_coarse, model_fine);

maskregion = false(size(labelmap));

for i = 1:length(parts)
    if strcmp(parts{i}, 'hair')
        index = 10;
    elseif strcmp(parts{i}, 'face')
        index = 20;
    elseif strcmp(parts{i}, 'upperBody')
        index = 30;
    elseif strcmp(parts{i}, 'lowerBody')
        index = 40;
    elseif strcmp(parts{i}, 'hands')
        index = 51;
    elseif strcmp(parts{i}, 'legs')
        index = 61;
    elseif strcmp(parts{i}, 'shoes')
        index = 63;
    else
        error('Body part invalid');
    end
    maskregion = (maskregion | (labelmap == index));
end

maskregion = resize_mask(maskregion, [size(img,1), size(img,2)]);
rects = cover_rects(maskregion, 9);

masked_img = img;
g = fspecial('gaussian', [15,15], 9);
for i = 1:size(rects, 2)
    x1 = rects(1, i);
    y1 = rects(2, i);
    x2 = rects(3, i);
    y2 = rects(4, i);
    tmp = double(masked_img(y1:y2, x1:x2, :));
    tmp = imfilter(tmp, g, 'same', 'replicate');
    masked_img(y1:y2, x1:x2, :) = uint8(tmp);
end
end

function resized_mask = resize_mask(mask, siz)

resized_mask = uint8(mask) * 255;
resized_mask = imresize(resized_mask, siz, 'nearest');
resized_mask(resized_mask < 128) = 0;
resized_mask = logical(resized_mask);
end