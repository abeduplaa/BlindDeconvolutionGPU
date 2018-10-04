#ifndef DOWN_CONVOLUTION_H
#define DOWN_CONVOLUTION_H


void computeDownConvolutionCPU(float *imgOut, const float *imgIn, const float *kernel, int n, int m, int nc, int w, int h);

void computeDownConvolutionGlobalMemCuda(float *imgOut, const float *imgIn, const float *kernel, const int n, const int m, const int nc, const int w, const int h);

#endif
