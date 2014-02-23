function visualize_bodyparts(dname)

load('colormap.mat');
load(fullfile('..', '..', 'data', 'attributes', [dname '_parse.mat']));

m = 3;
n = 6;

figure;
for i = 1:m*n
    subplot(m, n, i);
    
    j = randi(size(images, 1));
    nviews = size(images, 2);
    while isempty(images{j, nviews})
        nviews = nviews-1;
    end
    k = randi(nviews);
    
    img = images{j, k};
    bdparts = bodyparts{j, k};
    
    imshow(img);
    hold on;
    h = imshow(bdparts, colormap);
    hold off;
    set(h, 'AlphaData', (bdparts > 0) * 1);
end

end