function parse_attrreid_dataset(dname, model_coarse, model_fine)

load(fullfile('..', '..', 'data', 'attrreid', [dname '.mat']));

bodyparts = cell(size(images));

hwait = waitbar(0, '0%');
for i = 1:size(images, 1)
    for j = 1:size(images, 2)
        if isempty(images{i, j}); break; end;
        view_images = images{i, j};
        view_bodyparts = cell(size(view_images));
        for k = 1:length(view_images)
            view_images{k} = imresize(view_images{k}, [160 60]);
            view_bodyparts{k} = decompositionalLayer(view_images{k}, model_coarse, model_fine);
            view_bodyparts{k} = imresize(view_bodyparts{k}, [160 60], 'nearest');
        end
        images{i, j} = view_images;
        bodyparts{i, j} = view_bodyparts;
    end
    waitbar(i/size(images,1), hwait, sprintf('%d%%', ceil(i*100/size(images, 1))));
end

save(fullfile('..', '..', 'data', 'attrreid', [dname, '_parse.mat']), ...
    'images', 'bodyparts', 'attributes');

close(hwait);
clear images attributes;
