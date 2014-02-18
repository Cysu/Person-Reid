% This make.m is used under Windows

%add -largeArrayDims on 64-bit machines

mex -O -largeArrayDims -c svm.cpp
mex -O -largeArrayDims -c svm_model_matlab.cpp
mex -O svmtrain.cpp svm.o svm_model_matlab.o
mex -O svmpredict.cpp svm.o svm_model_matlab.o
mex -O libsvmread.cpp
mex -O libsvmwrite.cpp
