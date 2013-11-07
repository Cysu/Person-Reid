# Person-Reid #

Research projects on person re-identification


## Definition ##

The purpose of person re-identification is to recognize a unique pedestrian
among different camera views.

The pedestrian should have little discrepancy in appearance, and the cameras
should be still.

A typical input is two sets of pedestrians images sequences, namely *gallery*
and *probe*. The output should be a distance matrix. *Cumulative Match
Characteristic (CMC)* curve is used for evaluation.


## Data Format ##

We define a unified data format for person re-identification. The data is 
stored in Matlab file format, and the structure is illustrated below.

The top level variable (denoted as `D`) is a `d×1` cells array, where `d` is the 
number of different cameras-settings.

For each cameras-settings `D{t}`, the number of different camera views are 
fixed, as well as the camera parameters for each view. Hence, the structure of 
`D{t}` can be written as,

* `D{t}.params` (optionally, `v×1` cells array, each cell is a parameter struct)

* `D{t}.pedes` (`m×v` cells matrix, `m` is the number of pedestrians, each cell
is an images cells array)

A particular image is thus indexed by `D{t}.pedes{i, j}{k}`.

All floating numbers should be 32-bit.
