#ifndef DOWN_CONVOLUTION_H
#define DOWN_CONVOLUTION_H


void computeDownConvolutionCPU(float *imgOut, const float *imgIn, const float *kernel, int w, int h, int nc, int m, int n);

void computeDownConvolutionGlobalMemCuda(float *imgOut, const float *imgIn, const float *kernel, const int w, const int h, const int nc, const int m, const int n);

#endif
