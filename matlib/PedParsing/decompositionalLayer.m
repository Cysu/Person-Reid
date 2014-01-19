function [ labelmap ] = decompositionalLayer( img, model_coarse, model_fine )
% =========================================================================
%                 Pedestrian Parsing via Deep Decompositional Network
% =========================================================================
% This is a demo of the decompositional layer described in the following paper   
%
%   [1] P. Luo, X. Wang, and X. Tang, Pedestrian Parsing via Deep Decompositional Neural Network,
%   in Proceedings of IEEE International Conference on Computer Vision (ICCV) 2013 
% =========================================================================
% INPUTS:
%        img - input rgb image with size [160 60]
%        model_coarse - coarse scale model, in '.\model\'
%        model_fine - fine scale model, in '.\model\'
% OUTPUT:
%        labelmap - result of pedestrian parsing
% =========================================================================

if size(img,3) ~= 3
    error('Input image must has three channels!');
end
if size(img,1) ~= 160 || size(img,2) ~= 60
    error('Size of the input image must be [160,60]!');
end
indexmap =  [10, 20, 30, 51, 40, 61, 63];

% % coarse decomposition
feat = features(im2double(img), 5);
feat = feat(:)';
y = feat*model_coarse.W+model_coarse.b;
y = 1./(1+exp(-y));

labelmap = zeros(80, 30, 'uint8');
for j = 1 : 7
    tmp = reshape(y((j-1)*2400+1 : j*2400), [80,30]);
    labelmap(tmp>.3) = indexmap(j);
end

% % fine decomposition
feat = features(double(im2double(img).*repmat(double(imresize(labelmap, 2, 'nearest')), [1 1 3])), 3);
feat = feat(:)';
y = feat*model_fine.W+model_fine.b;
y = 1./(1+exp(-y));

labelmap = zeros(80, 30, 'uint8');
for j = 1 : 7
    tmp = reshape(y((j-1)*2400+1 : j*2400), [80,30]);
    labelmap(tmp>.3) = indexmap(j);
end

end

