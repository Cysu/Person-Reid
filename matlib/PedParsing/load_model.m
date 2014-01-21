load(fullfile('model', 'coarse_compressed.mat'));
model_coarse.W = Uk*full(Sk)*Vk'; model_coarse.b = b;
load(fullfile('model', 'fine_compressed.mat'));
model_fine.W = Uk*full(Sk)*Vk'; model_fine.b = b;
load(fullfile('model', 'clrmap.mat'));