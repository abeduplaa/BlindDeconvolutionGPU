#ifndef DOWN_CONVOLUTION_H
#define DOWN_CONVOLUTION_H

#include <iostream>

void computeDownConvolutionCPU(float *imgOut, const float *imgIn, const float *kernel, int w, int h, int nc, int m, int n);


#endif
