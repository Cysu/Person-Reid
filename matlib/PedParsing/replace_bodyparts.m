function replaced_img = replace_bodyparts(srcimg, tgtimg, parts, model_coarse, model_fine)

srcmap = decompositionalLayer(srcimg, model_coarse, model_fine);
tgtmap = decompositionalLayer(tgtimg, model_coarse, model_fine);

replaced_img = tgtimg;

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
    
    srcregion = (srcmap == index);
    tgtregion = (tgtmap == index);
    
    srcregion = resize_mask(srcregion, [size(srcimg,1), size(srcimg,2)]);
    tgtregion = resize_mask(tgtregion, [size(tgtimg,1), size(tgtimg,2)]);
    
    srcrects = cover_rects(srcregion, 9);
    tgtrects = cover_rects(tgtregion, 9);
    
    assert(size(srcrects, 2) == 1);
    assert(size(tgtrects, 2) == 1);
    
    replaced_img(tgtrects(2):tgtrects(4), tgtrects(1):tgtrects(3), :) = ...
        imresize(srcimg(srcrects(2):srcrects(4), srcrects(1):srcrects(3), :), ...
                 [tgtrects(4)-tgtrects(2)+1, tgtrects(3)-tgtrects(1)+1]);
end
end

function resized_mask = resize_mask(mask, siz)

resized_mask = uint8(mask) * 255;
resized_mask = imresize(resized_mask, siz, 'nearest');
resized_mask(resized_mask < 128) = 0;
resized_mask = logical(resized_mask);
end
