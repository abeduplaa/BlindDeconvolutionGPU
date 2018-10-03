#ifndef ADJUSTIMG_H
#define ADJUSTIMG_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void adjustImageSizeToOdd(cv::Mat& mIn, int& w, int& h, int& nc);

#endif
