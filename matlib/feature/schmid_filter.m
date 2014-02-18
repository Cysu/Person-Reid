function schmid = schmid_filter(sigma, tau)
%SCHMID_FILTER Create 2-D schmid filter.

% Compute the filter size
nstds = 3;
radius = ceil(max(1, nstds*sigma));
[x, y] = meshgrid(-radius:radius);

r = sqrt(x.^2 + y.^2);
schmid = cos(2*pi*tau*r/sigma) .* exp(-r.^2/(2*sigma^2));
schmid = schmid - mean(schmid(:));
schmid = schmid / norm(schmid, 1);