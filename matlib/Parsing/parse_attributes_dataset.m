function parse_attributes_dataset(dname, model_coarse, model_fine)

load(fullfile('..', '..', 'data', 'attributes', [dname '.mat']));

resized_images = cell(size(images));
bodyparts = cell(size(images));

hwait = waitbar(0, '0%');
for i = 1:size(images, 1)
    for j = 1:size(images, 2)
        if isempty(images{i, j}); break; end;
        resized_images{i, j} = imresize(images{i, j}, [160 60]);
        bodyparts{i, j} = decompositionalLayer(resized_images{i, j}, model_coarse, model_fine);
        bodyparts{i, j} = refine_feet(bodyparts{i, j});
        bodyparts{i, j} = imresize(bodyparts{i, j}, [160 60], 'nearest');
    end
    waitbar(i/size(images,1), hwait, sprintf('%d%%', ceil(i*100/size(images, 1))));
end

images = resized_images;
save(fullfile('..', '..', 'data', 'attributes', [dname, '_parse.mat']), ...
    'images', 'bodyparts', 'attributes');

close(hwait);
clear images attributes;

end
