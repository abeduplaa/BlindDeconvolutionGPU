#ifndef DOWN_CONVOLUTION_H
#define DOWN_CONVOLUTION_H


void computeDownConvolutionCPU(float *imgOut, const float *imgIn, const float *kernel, int w, int h, int nc, int m, int n);

computeDownConvolutionGlobalMemCuda(float *imgOut, const float *imgIn, const float *kernel, int w, int h, int nc, int m, int n);

#endif
