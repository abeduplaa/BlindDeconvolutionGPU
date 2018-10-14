// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#include "helper.cuh"

#include <cstdlib>
#include <iostream>
#include <sstream>

// cuda error checking
std::string prev_file = "";
int prev_line = 0;
void cuda_check(std::string file, int line) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::cout << std::endl << file << ", line " << line 
                  << ": " << cudaGetErrorString(e) << " (" << e << ")" << std::endl;
        if (prev_line > 0)
            std::cout << "Previous CUDA call:" << std::endl 
                      << prev_file << ", line " << prev_line << std::endl;
        exit(1);
    }
    prev_file = file;
    prev_line = line;
}


// OpenCV image conversion: layered to interleaved
void convertLayeredToInterleaved(float *aOut, const float *aIn, int w, int h, int nc)
{
    if (!aOut || !aIn)
    {
        std::cerr << "arrays not allocated!" << std::endl;
        return;
    }

    if (nc == 1)
    {
        memcpy(aOut, aIn, w*h*sizeof(float));
        return;
    }

    size_t nOmega = (size_t)w*h;
    for (int y=0; y<h; y++)
    {
        for (int x=0; x<w; x++)
        {
            for (int c=0; c<nc; c++)
            {
                aOut[(nc-1-c) + nc*(x + (size_t)w*y)] = aIn[x + (size_t)w*y + nOmega*c];
            }
        }
    }
}
void convertLayeredToMat(cv::Mat &mOut, const float *aIn)
{
    convertLayeredToInterleaved((float*)mOut.data, aIn, mOut.cols, mOut.rows, mOut.channels());
}


// OpenCV image conversion: interleaved to layered
void convertInterleavedToLayered(float *aOut, const float *aIn, int w, int h, int nc)
{
    if (!aOut || !aIn)
    {
        std::cerr << "arrays not allocated!" << std::endl;
        return;
    }

    if (nc == 1)
    {
        memcpy(aOut, aIn, w*h*sizeof(float));
        return;
    }

    size_t nOmega = (size_t)w*h;
    for (int y=0; y<h; y++)
    {
        for (int x=0; x<w; x++)
        {
            for (int c=0; c<nc; c++)
            {
                aOut[x + (size_t)w*y + nOmega*c] = aIn[(nc-1-c) + nc*(x + (size_t)w*y)];
            }
        }
    }
}

void convertMatToLayered(float *aOut, const cv::Mat &mIn) {
    convertInterleavedToLayered(aOut, (float*)mIn.data, mIn.cols, mIn.rows, mIn.channels());
}

// show cv:Mat in OpenCV GUI
// open camera using OpenCV
bool openCamera(cv::VideoCapture &camera, int device, int w, int h)
{
    if(!camera.open(device))
    {
        return false;
    }
    camera.set(CV_CAP_PROP_FRAME_WIDTH, w);
    camera.set(CV_CAP_PROP_FRAME_HEIGHT, h);
    return true;
}

// show cv:Mat in OpenCV GUI
void showImage(std::string title, const cv::Mat &mat, int x, int y)
{
    const char *wTitle = title.c_str();
    cv::namedWindow(wTitle, CV_WINDOW_AUTOSIZE);
    cv::moveWindow(wTitle, x, y);
    cv::imshow(wTitle, mat);
}

// show histogram in OpenCV GUI
void showHistogram256(const char *windowTitle, int *histogram, int windowX, int windowY)
{
    if (!histogram)
    {
        std::cerr << "histogram not allocated!" << std::endl;
        return;
    }

    const int nbins = 256;
    cv::Mat canvas = cv::Mat::ones(125, 512, CV_8UC3);

    float hmax = 0;
    for(int i = 0; i < nbins; ++i)
        hmax = max((int)hmax, histogram[i]);

    for (int j = 0, rows = canvas.rows; j < nbins-1; j++)
    {
        for(int i = 0; i < 2; ++i)
            cv::line(
                        canvas,
                        cv::Point(j*2+i, rows),
                        cv::Point(j*2+i, rows - (histogram[j] * 125.0f) / hmax),
                        cv::Scalar(255,128,0),
                        1, 8, 0
                        );
    }

    showImage(windowTitle, canvas, windowX, windowY);
    cv::imwrite("histogram.png",canvas*255.f);
}


// add Gaussian noise
float noise(float sigma)
{
    float x1 = (float)rand()/RAND_MAX;
    float x2 = (float)rand()/RAND_MAX;
    return sigma * sqrtf(-2*log(std::max(x1,0.000001f)))*cosf(2*M_PI*x2);
}

void addNoise(cv::Mat &m, float sigma)
{
    float *data = (float*)m.data;
    int w = m.cols;
    int h = m.rows;
    int nc = m.channels();
    size_t n = (size_t)w*h*nc;
    for(size_t i=0; i<n; i++)
    {
        data[i] += noise(sigma);
    }
}

// subtract 2 matrices:
void subtractArrays(float *arrayOut,const float *A, const float *B, const int size)
{
    //0. check that arrays are the same size

    //1. subtract A from B
    //TODO: REPLACE THIS WITH CUBLAS LIBRARY SUBTRACTION FUNCTION
    for(int i=0; i<size; i++)
    {
        arrayOut[i] = A[i] - B[i];
    }
}

#ifdef DEBUG
void saveMatrixMatlab(const char *key_name,
                      float *array,
                      int dim_x,
                      int dim_y,
                      int dim_z) {


    PyObject *pName, *pModule, *pFunc;
    PyObject *pArgs;

    // load module and function
    /*pName = PyString_FromString("pyfunctions");*/
    pName = PyUnicode_DecodeFSDefault("pyfunctions");
    pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule == NULL) {
        std::cout << "ERROR: cannot import scipy IO" << std::endl;
        PyErr_Print();
        exit(1);
    }

    // convert C-array to numpy array
    pFunc = PyObject_GetAttrString(pModule, "save_matrix");
    if (PyCallable_Check(pFunc) == 0) {
        std::cout << "ERROR: cannot link savemat function" << std::endl;
        PyErr_Print();
        exit(1);
    }

    if(PyArray_API == NULL) {
        _import_array();
    }

    const int num_dims = 3;
    npy_intp dims[num_dims] = {dim_z, dim_y, dim_x};
    PyObject *numpy_array = PyArray_SimpleNewFromData(num_dims,
                                                      dims,
                                                      NPY_FLOAT,
                                                      array);

    // create and init a dictionary to write data to a text file
    PyObject* key = PyUnicode_FromString(key_name); 

    // set up patameters for python function call
    pArgs = PyTuple_New(2);
    PyTuple_SetItem(pArgs, 0, key);
    PyTuple_SetItem(pArgs, 1, numpy_array);

    // call python
    PyObject *pOutput = PyObject_CallObject(pFunc, pArgs);
    if (pOutput == NULL) {
        std::cout << "cannot call save_matrix" << std::endl;
        PyErr_Print();
        exit(1);
    }

}
#endif
