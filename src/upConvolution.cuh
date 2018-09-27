#ifndef UP_CONVOLUTION_H
#define UP_CONVOLUTION_H

#include <iostream>

void initialiseKernel(float *kernel, int m, int n);

void computeUpConvolutionCPU(float *imgOut, const float *imgIn, const float *kernel, int w, int h, int nc, int m, int n);

/*void computeConvolutionSharedMemCuda(float *imgOut, const float *imgIn, const float *kernel, int kradius, int w, int h, int nc);*/
void computeUpConvolutionGlobalMemCuda(float *imgOut, const float *imgIn, const float *kernel, int w, int h, int nc, int m, int n);

#endif
