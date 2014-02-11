function replaced_img = replace_bodyparts(srcimg, tgtimg, parts, model_coarse, model_fine)

srcimg = imresize(srcimg, [160, 60]);
tgtimg = imresize(tgtimg, [160, 60]);

srcmap = decompositionalLayer(srcimg, model_coarse, model_fine);
tgtmap = decompositionalLayer(tgtimg, model_coarse, model_fine);

srcregion = false(size(srcmap));
tgtregion = false(size(tgtmap));

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
    
    srcregion = (srcregion | (srcmap == index));
    tgtregion = (tgtregion | (tgtmap == index));
end

srcregion = resize_mask(srcregion, [size(srcimg,1), size(srcimg,2)]);
tgtregion = resize_mask(tgtregion, [size(tgtimg,1), size(tgtimg,2)]);

srcrects = cover_rects(srcregion, 9);
tgtrects = cover_rects(tgtregion, 9);

if size(srcrects, 2) < 1 || size(tgtrects, 2) < 1
    replaced_img = [];
    return;
else
    replaced_img = tgtimg;
end

if size(srcrects, 2) ~= 1
    srcrects = choose_max_rect(srcrects);
end
if size(tgtrects, 2) ~= 1
    tgtrects = choose_max_rect(tgtrects);
end

assert(size(srcrects, 2) == 1);
assert(size(tgtrects, 2) == 1);

[srcrects, tgtrects] = coherent_rect(srcrects, tgtrects, ...
    [size(srcimg,1), size(srcimg,2)], [size(tgtimg,1), size(tgtimg,2)]);

replaced_img(tgtrects(2):tgtrects(4), tgtrects(1):tgtrects(3), :) = ...
    imresize(srcimg(srcrects(2):srcrects(4), srcrects(1):srcrects(3), :), ...
             [tgtrects(4)-tgtrects(2)+1, tgtrects(3)-tgtrects(1)+1]);

end

function resized_mask = resize_mask(mask, siz)

resized_mask = uint8(mask) * 255;
resized_mask = imresize(resized_mask, siz, 'nearest');
resized_mask(resized_mask < 128) = 0;
resized_mask = logical(resized_mask);
end

function rect = choose_max_rect(rects)

maxarea = 0;
maxi = 0;

for i = 1:size(rects, 2)
    w = rects(3)-rects(1)+1;
    h = rects(4)-rects(2)+1;
    if w*h > maxarea
        maxarea = w*h;
        maxi = i;
    end
end

rect = rects(:, maxi);
end

function [src, tgt] = coherent_rect(src, tgt, srcimsiz, tgtimsiz)

rect_w = max(src(3)-src(1)+1, tgt(3)-tgt(1)+1);
rect_h = max(src(4)-src(2)+1, tgt(4)-tgt(2)+1);

src = stretch_rect(src, srcimsiz, [rect_h, rect_w]);
tgt = stretch_rect(tgt, tgtimsiz, [rect_h, rect_w]);
end

function rect = stretch_rect(rect, imsiz, stretchsiz)

cx = floor((rect(3)+rect(1))/2);
cy = floor((rect(4)+rect(2))/2);

x1 = cx - floor(stretchsiz(2)/2);
y1 = cy - floor(stretchsiz(1)/2);
x2 = x1 + stretchsiz(2) - 1;
y2 = y1 + stretchsiz(1) - 1;

if x1 < 1
    dx = 1 - x1;
elseif x2 > imsiz(2)
    dx = imsiz(2) - x2;
else
    dx = 0;
end

if y1 < 1
    dy = 1 - y1;
elseif y2 > imsiz(1)
    dy = imsiz(1) - y2;
else
    dy = 0;
end

rect(1) = x1 + dx;
rect(3) = x2 + dx;
rect(2) = y1 + dy;
rect(4) = y2 + dy;
end
