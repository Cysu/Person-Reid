function gabor = gabor_filter(gamma, theta, lambda, sigma, psi)
%GABOR_FILTER Create 2-D complex gabor filter.

sigma_x = sigma;
sigma_y = sigma / gamma;

% Compute the filter size
nstds = 3;
xmax = max(abs(nstds*sigma_x*cos(theta)),abs(nstds*sigma_y*sin(theta)));
xmax = ceil(max(1,xmax));
ymax = max(abs(nstds*sigma_x*sin(theta)),abs(nstds*sigma_y*cos(theta)));
ymax = ceil(max(1,ymax));
radius = max(ymax, xmax);
[x, y] = meshgrid(-radius:radius, -radius:radius);

xprime = x*cos(theta) + y*sin(theta);
yprime = -x*sin(theta) + y*cos(theta);

gabor = exp(-0.5*(xprime.^2/sigma_x^2 + yprime.^2/sigma_y^2)).* ...
    exp(1i * (2*pi/lambda*xprime + psi));
