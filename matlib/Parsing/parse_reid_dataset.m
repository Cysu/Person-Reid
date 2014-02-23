function parse_reid_dataset(dname, model_coarse, model_fine)

load(fullfile('..', '..', 'data', 'reid', [dname '.mat']));

% Stack all the pedestrains in to one array
images = cell(20000, 2);
count = 0;

for i = 1:size(data, 1)
    pedes = data(i).pedes;
    assert(size(pedes, 2) == 2);  % Should be only two views
    for j = 1:size(pedes, 1)
        count = count + 1;
        for v = 1:2
            images{count, v} = pedes{j, v};
        end
    end
end

images = images(1:count, :);

% Parse every image
bodyparts = cell(size(images));

m = size(images, 1);
hwait = waitbar(0, '0%');
for i = 1:m
    for v = 1:size(images, 2)
        view = images{i, v};
        result = cell(size(view));
        for k = 1:length(view)
            view{k} = imresize(view{k}, [160 60]);
            result{k} = ...
                imresize(...
                    refine_feet(...
                        decompositionalLayer(view{k}, model_coarse, model_fine)), ...
                    [160 60], ...
                    'nearest' ...
               );
        end
        images{i, v} = view;
        bodyparts{i, v} = result;
    end
    waitbar(i/m, hwait, sprintf('%.2f%%', i*100/m));
end
close(hwait);

% Save the parsing result
save(fullfile('..', '..', 'data', 'reid', [dname, '_parse.mat']), ...
    'images', 'bodyparts');

clear data;
