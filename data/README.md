## Data Format ##

We define a unified data format for person re-identification. The data is 
stored in Matlab file format, and the structure is illustrated below.

The top level variable (denoted as `data`) is a `d×1` structs array, where `d` 
is the number of different groups of cameras-settings.

For each group `data(t)`, the number of different camera views are fixed, as 
well as the camera parameters for each view. Hence, the structure of `data(t)`
can be written as,

* `data(t).params` (optionally, `v×1` cells array, each cell is a parameter struct)

* `data(t).pedes` (`m×v` cells matrix, `m` is the number of pedestrians, each cell
is an images cells array)

A particular image is thus indexed by `data(t).pedes{i, j}{k}`.

All floating numbers should be 32-bit.
