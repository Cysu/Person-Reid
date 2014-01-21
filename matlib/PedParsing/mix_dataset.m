function [miximages, mixattributes] = mix_dataset(dataset_name, attr_inds, model_coarse, model_fine)

load(fullfile('..', '..', 'data', 'attributes', [dataset_name '.mat']));

npedes = size(images, 1);
nviews = size(images, 2);

subattrs = false(length(attr_inds), npedes);
for i = 1:npedes
    tmp = attributes{i,1};
    subattrs(:, i) = tmp(attr_inds);
end

miximages = cell(npedes*nviews, 1);
mixattributes = cell(npedes*nviews, 1);
count = 0;

for i = 1:npedes
    for j = 1:nviews
        if isempty(images{i, j}); break; end;
        fprintf('%d %d\n', i, j);
        while true
            p = randi(npedes);
            q = randi(nviews);
            if isempty(images{p, q}); continue; end;
            if isequal(subattrs(:, i), subattrs(:, p)); continue; end;
            img = images{i,j};
            img = replace_bodyparts(images{p,q}, img, {'hair', 'face'}, ...
                model_coarse, model_fine);
            if isempty(img); break; end;
            count = count + 1;
            miximages{count, 1} = img;
            mixattributes{count, 1} = attributes{p,1};
            break;
        end
    end
end

images = miximages(1:count, :);
attributes = mixattributes(1:count, :);

save(fullfile('..', '..', 'data', 'attributes', [dataset_name '_mix.mat']), ...
    'images', 'attributes');

end