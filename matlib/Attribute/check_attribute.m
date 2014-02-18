function valid = check_attribute(attribute)

global unival;
global multival;
global names;

valid = true;

for i = 1:length(unival)
    grp = unival{i};
    subattr = zeros(length(grp), 1);
    for j = 1:length(grp)
        k = ismember(names, grp{j});
        subattr(j) = attribute(k);
    end
    if sum(subattr) ~= 1; valid = false; return; end;
end

for i = 1:length(multival)
    grp = multival{i};
    subattr = zeros(length(grp), 1);
    for j = 1:length(grp)
        k = ismember(names, grp{j});
        subattr(j) = attribute(k);
    end
    if sum(subattr) == 0; valid = false; return; end;
end

end