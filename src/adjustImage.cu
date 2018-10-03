#include "adjustImage.cuh"

void adjustImageSizeToOdd(cv::Mat& mIn, int& w, int& h, int& nc){
    if(w % 2 == 0)
        --w;
    if(h % 2 == 0)
        --h;
    cv::Rect myRectangle(0, 0, w, h);
    cv::Mat temp =  mIn(myRectangle);
    temp.copyTo(mIn);
}
