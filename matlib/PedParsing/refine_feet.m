function fine = refine_feet(coarse)

strip = 66:80;
img = coarse(strip, :);

bin = (img > 0);
se = strel('disk', 3);
bin = imdilate(bin, se);
bin = xor(bin, ((img > 0) & (img ~= 70)));
se = strel('disk', 1);
bin = imerode(bin, se);

img(bin) = 70;
fine = coarse;
fine(strip, :) = img;

end