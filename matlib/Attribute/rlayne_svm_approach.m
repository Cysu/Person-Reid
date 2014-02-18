%% Load dataset

run('attrconf.m');

m = 0;
for t = 1:length(datasets)
    load(fullfile('..', '..', 'data', 'attributes', [datasets{t} '.mat']));
    for i = 1:size(images, 1)
        if ~check_attribute(attributes{i}); continue; end;
        for j = 1:size(images, 2)
            if isempty(images{i, j}); break; end;
            m = m + 1;
            imgs{m} = images{i, j};
            attrs{m} = attributes{i};
        end
    end
end

save('rlayne_dataset.mat', 'imgs', 'attrs');

%% Extract feautres
if exist('rlayne_dataset.mat', 'file') && (~exist('imgs', 'var') || ~exist('attrs', 'var'))
    load('rlayne_dataset.mat');
end

n_feat = 2784;
n_attr = length(attrs{1});

gabor = {
    gabor_filter(0.3, 0, 4, 2, 0), ...
    gabor_filter(0.3, 0, 8, 2, 0), ...
    gabor_filter(0.4, 0, 4, 1, 0), ...
    gabor_filter(0.4, 0, 8, 1, 0), ...
    gabor_filter(0.3, pi/2, 4, 1, 0), ...
    gabor_filter(0.3, pi/2, 8, 2, 0), ...
    gabor_filter(0.4, pi/2, 4, 1, 0), ...
    gabor_filter(0.4, pi/2, 8, 2, 0)
};
schmid = {
    schmid_filter(2, 1), ...
    schmid_filter(4, 1), ...
    schmid_filter(4, 2), ...
    schmid_filter(6, 1), ...
    schmid_filter(6, 2), ...
    schmid_filter(6, 3), ...
    schmid_filter(8, 1), ...
    schmid_filter(8, 2), ...
    schmid_filter(8, 3), ...
    schmid_filter(10, 1), ...
    schmid_filter(10, 2), ...
    schmid_filter(10, 3)    
};

X = zeros(n_feat, m);
Y = zeros(n_attr, m);

hwait = waitbar(0, 'Extract features 0%%');
for i = 1:m
    X(:, i) = rlayne_extract_feature(imgs{i}, gabor, schmid);
    Y(:, i) = logical(attrs{i});
    waitbar(i/m, hwait, sprintf('Extract features %d%%', int32(i*100/m)));
end
close(hwait);

inds = randperm(m);
X = X(:, inds);
Y = Y(:, inds);

n_train = ceil(m * 0.7);
train_X = X(:, 1:n_train);
train_Y = Y(:, 1:n_train);
test_X = X(:, n_train+1:m);
test_Y = Y(:, n_train+1:m);

save('rlayne_feature.mat', 'train_X', 'train_Y', 'test_X', 'test_Y');

%% Train a SVM for each attribute
if exist('rlayne_feature.mat', 'file') && (~exist('X', 'var') || ~exist('Y', 'var'))
    load('rlayne_feature.mat');
end

models = cell(n_attr, 1);
for i = 1:n_attr
    fprintf('Training %d attribute: %s\n', i, names{i});
    models{i} = svmtrain(train_Y(i, :)', train_X', '-t 5 -b 1 -q');
end

save('rlayne_model.mat', 'models');

%% Test the SVM models
if exist('rlayne_model.mat', 'file') && ~exist('models', 'var')
    load('rlayne_model.mat');
end

m = size(test_X, 2);
for i = 1:length(unival)
    fprintf('%s,frequency,accuracy\n', unival_titles{i});

    grp = unival{i};
    outputs = zeros(length(grp), m);
    targets = zeros(length(grp), m);
    for j = 1:length(grp)
        k = find(ismember(names, grp{j}));
        targets(j, :) = test_Y(k, :);
        
        if models{k}.nr_class ~= 2
            outputs(j, :) = targets(j, :);
            continue;
        end
        
        [label, ~, prob] = svmpredict(test_Y(k, :)', test_X', models{k}, '-b 1');
        if (label(1) == 1 && prob(1,1) > prob(1,2)) || ...
                (label(1) == 0 && prob(1,1) < prob(1,2))
            outputs(j, :) = prob(:, 1)';
        else
            outputs(j, :) = prob(:, 2)';
        end
    end

    [~, targets] = max(targets, [], 1);
    [~, outputs] = max(outputs, [], 1);

    freqs = zeros(length(grp), 1);
    accs = zeros(length(grp), 1);
    for j = 1:length(grp)
        freqs(j) = mean(targets == j);
        accs(j) = sum((targets == j) & (outputs == j)) / sum(targets == j);
        if isnan(accs(j)); accs(j) = 0; end;
        fprintf('%s,%f,%f\n', grp{j}, freqs(j), accs(j));
    end
    fprintf('Overall accuracy = %f\n\n', sum(freqs.*accs));

end

for i = 1:length(multival)
    fprintf('%s,frequency,TPR,FPR\n', multival_titles{i});
    
    grp = multival{i};
    outputs = zeros(length(grp), m);
    targets = zeros(length(grp), m);
    for j = 1:length(grp)
        k = find(ismember(names, grp{j}));
        targets(j, :) = test_Y(k, :);
        
        if models{k}.nr_class ~= 2
            outputs(j, :) = targets(j, :);
            continue;
        end
        
        [label, ~, ~] = svmpredict(test_Y(k, :)', test_X', models{k}, '-b 1');
        outputs(j, :) = label';
    end
    
    freqs = zeros(length(grp), 1);
    tprs = zeros(length(grp), 1);
    fprs = zeros(length(grp), 1);
    for j = 1:length(grp)
        t = targets(j, :);
        o = outputs(j, :);
        freqs(j) = mean(t == 1);
        tprs(j) = sum((t == 1) & (o == 1)) / sum(t == 1);
        fprs(j) = sum((t == 0) & (o == 1)) / sum(t == 0);
        if isnan(tprs(j)); tprs(j) = 0; end;
        if isnan(fprs(j)); fprs(j) = 0; end;
        fprintf('%s,%f,%f,%f\n', grp{j}, freqs(j), tprs(j), fprs(j));
    end
    fprintf('Overall TPR = %f, FPR = %f\n\n', ...
        sum(freqs.*tprs) / sum(freqs), ...
        sum(freqs.*fprs) / sum(freqs));
end
