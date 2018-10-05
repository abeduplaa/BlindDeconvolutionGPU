#ifndef DOWN_CONVOLUTION_H
#define DOWN_CONVOLUTION_H


void computeDownConvolutionCPU(float *imgOut, const float *imgIn, const float *kernel, 
                               int n, int m, int nc, int w, int h);


void computeDownConvolutionGlobalMemCuda(float *imgOut, const float *imgIn, const float *kernel, 
                                         const int n, const int m, const int nc, 
                                         const int w, const int h);


void computeImageConvilution(float *d_kernel_temp, const int mk, const int nk ,
                             const float *d_imgIn, float *d_imgInBuffer, 
                             const int w, const int h, 
                             const float *d_imgInPad, const int padw, const int padh, 
                             const int nc); 

#endif
