// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#ifndef TUM_HELPER_CUH
#define TUM_HELPER_CUH

#include <cuda_runtime.h>
#include <ctime>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include "cublas_v2.h"
#include <cudnn.h>

#ifdef DEBUG
    #include <Python.h>
    #include <numpy/arrayobject.h>
#endif

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }


// CUDA utility functions

// cuda error checking
#define CUDA_CHECK cuda_check(__FILE__,__LINE__)
void cuda_check(std::string file, int line);


// compute index within 1d array 
inline __host__ __device__ int getIndex(int i, int j, int width) {
    return i + j * width;
}



// compute grid size from block size
inline dim3 computeGrid1D(const dim3 &block, const int w) {
    int num_blocks_x = (w + block.x - 1) / block.x;
    return dim3(num_blocks_x, 1, 1);
}

inline dim3 computeGrid2D(const dim3 &block, const int w, const int h) {
    int num_blocks_x = (w + block.x - 1) / block.x;
    int num_blocks_y = (h + block.y - 1) / block.y;
    return dim3(num_blocks_x, num_blocks_y, 1);
}

inline dim3 computeGrid3D(const dim3 &block, const int w, const int h, const int s) {
    int num_blocks_x = (w + block.x - 1) / block.x;
    int num_blocks_y = (h + block.y - 1) / block.y;
    int num_blocks_z = (s + block.z - 1) / block.z;
    return dim3(num_blocks_x, num_blocks_y, num_blocks_z); }


// OpenCV image conversion
// interleaved to layered
void convertMatToLayered(float *aOut, const cv::Mat &mIn);

// layered to interleaved
void convertLayeredToMat(cv::Mat &mOut, const float *aIn);


// OpenCV GUI functions
// open camera
bool openCamera(cv::VideoCapture &camera, int device, int w = 640, int h = 480);

// show image
void showImage(std::string title, const cv::Mat &mat, int x, int y);

// show histogram
void showHistogram256(const char *windowTitle, int *histogram, int windowX, int windowY);


// adding Gaussian noise
void addNoise(cv::Mat &m, float sigma);

// subtracting two arrays
void subtractArrays(float *arrayOut,const float *A, const float *B, const int size);

// measuring time
class Timer
{
public:
	Timer() : tStart(0), running(false), sec(0.f)
	{
	}
	void start()
	{
        cudaDeviceSynchronize();
		tStart = clock();
		running = true;
	}
	void end()
	{
		if (!running) { sec = 0; return; }
        cudaDeviceSynchronize();
		clock_t tEnd = clock();
		sec = (float)(tEnd - tStart) / CLOCKS_PER_SEC;
		running = false;
	}
	float get()
	{
		if (running) end();
		return sec;
	}
private:
	clock_t tStart;
	bool running;
	float sec;
};


#ifdef DEBUG
void saveMatrixMatlab(const char *key_name,
                      float *array,
                      int dim_x,
                      int dim_y,
                      int dim_z); 
#endif

#endif
