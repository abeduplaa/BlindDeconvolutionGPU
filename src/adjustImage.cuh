#ifndef ADJUSTIMG_H
#define ADJUSTIMG_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "helper.cuh"

void adjustImageSizeToOdd(cv::Mat& mIn, int& w, int& h, int& nc);


void rotateKernel_180(float *d_kernel_rot_180,
                      const float *d_kernel, 
                      const int mk, 
                      const int nk); 


void stackData(float *d_data, 
               const int w, 
               const int h,
               const int nc); 
#endif
