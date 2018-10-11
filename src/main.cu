// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################

#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdlib.h>
#include <string>

#include "test.cuh"
#include "helper.cuh"
#include "downConvolution.cuh"
#include "testdownConv.cuh"
#include "divergence.cuh"
#include "adjustImage.cuh"
#include "pad.cuh"
#include "upConvolution.cuh"
#include "epsilon.cuh"
#include "selectNonZero.cuh"
#include "normalise.cuh"
#include <fstream>

#include <Python.h>
#include <numpy/arrayobject.h>


#include "cublas_v2.h"

void saveMatrixMatlab(char *key_name,
                      float *array,
                      int dim_x,
                      int dim_y,
                      int dim_z) {

    PyObject *pName, *pModule, *pFunc;
    PyObject *pArgs;

    // load module and function
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

void pythonConv2(float *result,
                 float *input, int input_width, int input_height, int input_channels,
                 float *kernel, int kernel_width, int kernel_height, int kernel_channels,
                 PyObject *function, char *conv_mode, char *conv_tag) {
 
    int num_dim = 3;

    npy_intp image_dims[3] = {input_channels, input_height, input_width};
    
    PyObject *numpy_image = PyArray_SimpleNewFromData(num_dim,
                                                      image_dims,
                                                      NPY_FLOAT,
                                                      input);
    

    npy_intp kernel_dims[3] = {kernel_channels, kernel_height, kernel_width};
    
    PyObject *numpy_kernel = PyArray_SimpleNewFromData(num_dim,
                                                       kernel_dims,
                                                       NPY_FLOAT,
                                                       kernel);

    PyObject *mode = PyUnicode_DecodeFSDefault(conv_mode);
    PyObject *tag= PyUnicode_DecodeFSDefault(conv_tag);

    PyObject *params = PyTuple_New(4);
    PyTuple_SetItem(params, 0, numpy_image);
    PyTuple_SetItem(params, 1, numpy_kernel);
    PyTuple_SetItem(params, 2, mode);
    PyTuple_SetItem(params, 3, tag);

    if (PyCallable_Check(function) == 0) {
        std::cout << "ERROR: cannot link python_callback function" << std::endl;
        PyErr_Print();
        exit(1);
    }

    PyObject *pOutput = PyObject_CallObject(function, params);
    if (pOutput == NULL) {
        std::cout << "cannot call the users script" << std::endl;
        PyErr_Print();
        exit(1);
    }

    float *output = (float*)PyArray_DATA(pOutput);
    

    // copy results back to application from python
    int output_num_channels = kernel_channels == 3 ? 1 : 3;
    int output_width = -1, output_height = - 1;
    if (!strcmp(conv_mode, "valid")) {
        output_width = input_width - kernel_width + 1; 
        output_height = input_height - kernel_height + 1; 
    }
    else if (!strcmp(conv_mode, "full")) {
        output_width = input_width + kernel_width - 1; 
        output_height = input_height + kernel_height - 1;
    }
    else {
        std::cout << "THERE IS NO SUCH A MODE" << std::endl;
        exit(1);
    }

    for (int channel = 0; channel < output_num_channels; ++channel) {

        int output_offset = output_width * output_height * channel;
        for (int j = 0; j < output_height; ++j) {
            for (int i = 0; i < output_width; ++i) {
                result[i + j * output_width + output_offset] = 
                    output[i + j * output_width + output_offset];
            }
        }
    } 
}


int main(int argc,char **argv) {

    // TODO: ADD COMMAND LINE FUNCTIONS LATER

    Py_Initialize();

    PyObject *script_name = PyUnicode_DecodeFSDefault("pyfunctions");
    PyObject *py_script = PyImport_Import(script_name);

    if (py_script == NULL) {
        std::cout << "ERROR: cannot import the users script" << std::endl;
        PyErr_Print();
        exit(1);
    }

    PyObject *python_callback = PyObject_GetAttrString(py_script, "conv2");
    if (PyCallable_Check(python_callback) == 0) {
        std::cout << "ERROR: cannot link python_callback function" << std::endl;
        PyErr_Print();
        exit(1);
    }

    import_array();
    PyObject *pOutput;


    // parse command line parameters
    const char *params = {
        "{image| |input image}"
        "{bw|false|load input image as grayscale/black-white}"
        "{mk|5|kernel width }"
        "{nk|5|kernel height}"
        "{cpu|false|compute on CPU}"
        "{eps|1e-3| epsilon }"
        "{lambda|3.5e-4| lambda }"
        "{iter|1| iter}"
       // "{m|mem|0|memory: 0=global, 1=shared, 2=texture}"
    };
    cv::CommandLineParser cmd(argc, argv, params);

    // input image
    std::string inputImage = cmd.get<std::string>("image");
    // number of computation repetitions to get a better run time measurement
    // size_t repeats = (size_t)cmd.get<int>("repeats");
    // load the input image as grayscale
     bool gray = cmd.get<bool>("bw");
	 int mk = cmd.get<int>("mk"); mk = (mk <= 0) ? 5 : mk;
	 int nk = cmd.get<int>("nk"); nk = (nk <= 0) ? 5 : nk;
     bool is_cpu = cmd.get<bool>("cpu");
     float lambda = cmd.get<float>("lambda"); lambda = (lambda <= 0) ? 3.5e-4 : lambda; 
     float lambda_min = 0.0006f;
     float eps = cmd.get<float>("eps"); eps = ( eps <= 0 ) ? 1e-3 : eps;
     int iter = cmd.get<int>("iter"); iter = ( iter <= 0 ) ? 1 : iter;

     printf("EPS: %10.8f\n", eps);
     std::cout << "mode: " << (is_cpu ? "CPU" : "GPU") << std::endl;

    // TODO: LOAD IMAGE
    // read input frame
    cv::Mat mIn;
    // load the input image using opencv (load as grayscale if "gray==true", 
    // otherwise as is (may be color or grayscale))
    mIn = cv::imread(inputImage.c_str(), (gray ? CV_LOAD_IMAGE_GRAYSCALE : -1));

    // check
    if (mIn.empty()) {
        std::cerr << "ERROR: Could not retrieve frame " << inputImage << std::endl;
        return 1;
    }
    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn, CV_32F);

    // get image dimensions
    int w = mIn.cols;         // width
    int h = mIn.rows;         // height
    int nc = mIn.channels();  // number of channels
    std::cout << "Original Image: " << w << " x " << h  << " x " << nc << std::endl;
    adjustImageSizeToOdd(mIn, w, h, nc);

	int img_size = w * h * nc;
    int padw = w + mk - 1;
    int padh = h + nk - 1;
    int pad_img_size = padw * padh * nc;
    std::cout << "Image after Cropping: " << w << " x " << h  << " x " << nc << std::endl;
    std::cout << "Pad Image size" << padw << " x " << padh << " x " << nc << std::endl;

    // init kernel
    size_t kn = mk*nk;
    float *kernel = new float[kn * sizeof(float)];
      /*initialize kernel to uniform.*/
    /*initialiseKernel(kernel, mk, nk);*/
    std::cout << "MY INIT KERNEL:" << std::endl;
    for (int j = 0; j < nk; ++j) { 
        for (int i = 0; i < mk; ++i) {
            kernel[i + j * mk] = (double)(1 + i + j * mk) / kn;
            std::cout << kernel[i + j * mk] << "  ";
        }
        std::cout << std::endl;
    }

    // initialize CUDA context
    // cudaDeviceSynchronize();

    // Initialize Cublas
    cublasHandle_t handle;
    cublasCreate(&handle);

    // ### Set the output image format
    cv::Mat mOut(h, w, mIn.type());  // grayscale or color depending on input image, nc layers

    // ### Allocate arrays
    float *imgIn = new float[img_size];
    float *imgInPad = new float[pad_img_size]; //TODO: check size (RAVI)
    float *imgOut = new float[pad_img_size];
    float *dx_fw = new float[pad_img_size];
    float *dy_fw = new float[pad_img_size];
    float *dx_bw = new float[pad_img_size];
    float *dy_bw = new float[pad_img_size];
    float *dx_mixed = new float[pad_img_size];
    float *dy_mixed = new float[pad_img_size];
    float *imgDownConv0 = new float[img_size];
    float *imgUpConv = new float[pad_img_size];

    float *div = new float[pad_img_size];
    float epsU, epsK = 0.0f;
    float alpha = -1.0f;

    // Python
    float *scp_kernel = new float [pad_img_size];
    float *scp_image = new float [pad_img_size];
    float *scp_output = new float [pad_img_size];

    // TODO: ALLOCATE MEMORY ON GPU
    // allocate arrays on GPU
    float *d_imgIn = NULL;
    float *d_imgInPad = NULL;
    float *d_imgInBuffer = NULL;
    float *d_imgPadRot = NULL;
    float *d_imgOut = NULL;

    float *d_dx_fw = NULL;
    float *d_dy_fw = NULL;
    float *d_dx_bw = NULL;
    float *d_dy_bw = NULL;
    float *d_dx_mixed = NULL;
    float *d_dy_mixed = NULL;

    float *d_imgDownConv0 = NULL;
    float *d_imgDownConv1 = NULL;
    float *d_imgUpConv = NULL;

    float *d_div = NULL;
    float *d_kernel = NULL;
    float *d_kernel_temp = NULL;

    float *d_epsU = NULL;
    float *d_epsK = NULL;

    cudaMalloc(&d_imgIn, img_size * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_imgInBuffer, img_size * sizeof(float)); CUDA_CHECK;

    // TODO: be sure the size id right (RAVI)
    cudaMalloc(&d_imgInPad, pad_img_size * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_imgPadRot, pad_img_size * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_imgOut , pad_img_size * sizeof(float)); CUDA_CHECK;

    cudaMalloc(&d_kernel, kn * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_kernel_temp, kn  * sizeof(float)); CUDA_CHECK;

    cudaMalloc(&d_dx_fw, pad_img_size * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_dy_fw, pad_img_size * sizeof(float)); CUDA_CHECK;

    cudaMalloc(&d_dx_bw, pad_img_size * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_dy_bw, pad_img_size * sizeof(float)); CUDA_CHECK;

    cudaMalloc(&d_dx_mixed, pad_img_size * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_dy_mixed, pad_img_size * sizeof(float)); CUDA_CHECK;

    cudaMalloc(&d_div , pad_img_size * sizeof(float)); CUDA_CHECK;

    cudaMalloc(&d_imgDownConv0, img_size * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_imgDownConv1, img_size * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_imgUpConv, pad_img_size * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_epsU, sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_epsK, sizeof(float)); CUDA_CHECK;

	mIn /= 255.0f;
	convertMatToLayered(imgIn, mIn);
    cudaMemcpy(d_imgIn, imgIn, img_size * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
    cudaMemcpy(d_kernel, kernel, kn * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;

    padImgGlobalMemCuda(d_imgInPad, d_imgIn, w, h, nc, mk, nk);

    for(int iterations = 0; iterations < iter; ++iterations){

        /*padImgGlobalMemCuda(d_imgInPad, d_imgIn, w, h, nc, mk, nk);*/
    
        std::cout << "Iteration num:  " << iterations << std::endl;
        //$$$$$$$$$$$$$$$$$$ PATCH %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        /*rotateKernel_180(d_kernel_temp, d_kernel, mk, nk);*/
        /*computeDownConvolutionGlobalMemCuda(d_imgDownConv0, */
                                            /*d_imgInPad, */
                                            /*d_kernel_temp, */
                                            /*padw, */
                                            /*padh, */
                                            /*nc, */
                                            /*mk, nk);*/

        cudaMemcpy(scp_kernel, d_kernel, kn * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(scp_image, d_imgInPad, pad_img_size * sizeof(float), cudaMemcpyDeviceToHost);

        /*cudaMemcpy(scp_output, d_imgDownConv0, img_size * sizeof(float), cudaMemcpyDeviceToHost);*/
        /*saveMatrixMatlab("cuda_downconv_0", scp_output, w, h, nc);*/

        pythonConv2(scp_output, 
                    scp_image, padw, padh, nc,
                    scp_kernel, mk, nk, 1,
                    python_callback, 
                    "valid",
                    "downconv_0");
        PyErr_Print();

        cudaMemcpy(d_imgDownConv0,
                   scp_output, 
                   img_size * sizeof(float), cudaMemcpyHostToDevice);

        //$$$$$$$$$$$$$$$$$$ PATCH %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


        // DONE: cublas subtract k(+)*u - f. Move that to a separate function

        alpha = -1.0f;
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
        cublasSaxpy(handle, img_size, &alpha, d_imgIn, 1, d_imgDownConv0, 1); CUDA_CHECK;
        
        // TODO: compute(mirror, rotate) kernel
        rotateKernel_180(d_kernel_temp, d_kernel, mk, nk); 

        // TODO: check the list of  parameters 
        //$$$$$$$$$$$$$$$$$$ PATCH %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        /*computeUpConvolutionGlobalMemCuda(d_imgUpConv, d_imgDownConv0, d_kernel,*/
                                          /*w, h, nc, mk, nk);*/

        cudaMemcpy(scp_kernel, d_kernel_temp, kn * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(scp_image, d_imgDownConv0, img_size * sizeof(float), cudaMemcpyDeviceToHost);


        pythonConv2(scp_output, 
                    scp_image, w, h, nc,
                    scp_kernel, mk, nk, 1,
                    python_callback, 
                    "full",
                    "upconv");
        PyErr_Print();

        cudaMemcpy(d_imgUpConv,
                   scp_output, 
                   pad_img_size * sizeof(float), cudaMemcpyHostToDevice);
        //$$$$$$$$$$$$$$$$$$ PATCH %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        // compute gradient and divergence
        computeDiffOperatorsCuda(d_div, 
                                 d_dx_fw, d_dy_fw,
                                 d_dx_bw, d_dy_bw,
                                 d_dx_mixed, d_dy_mixed, 
                                 d_imgInPad, padw, padh, nc,  eps);
        
        // divergence
        cudaMemcpy(scp_kernel, d_div, pad_img_size * sizeof(float), cudaMemcpyDeviceToHost);
        saveMatrixMatlab("div_cuda", scp_kernel, padw, padh, nc);
        PyErr_Print();

        // gradients
        cudaMemcpy(scp_kernel, d_dx_fw, pad_img_size * sizeof(float), cudaMemcpyDeviceToHost);
        saveMatrixMatlab("cuda_fxforw", scp_kernel, padw, padh, nc);
        PyErr_Print();

        cudaMemcpy(scp_kernel, d_dy_fw, pad_img_size * sizeof(float), cudaMemcpyDeviceToHost);
        saveMatrixMatlab("cuda_fyforw", scp_kernel, padw, padh, nc);
        PyErr_Print();

        cudaMemcpy(scp_kernel, d_dx_bw, pad_img_size * sizeof(float), cudaMemcpyDeviceToHost);
        saveMatrixMatlab("cuda_fxback", scp_kernel, padw, padh, nc);
        PyErr_Print();

        cudaMemcpy(scp_kernel, d_dy_bw, pad_img_size * sizeof(float), cudaMemcpyDeviceToHost);
        saveMatrixMatlab("cuda_fyback", scp_kernel, padw, padh, nc);
        PyErr_Print();

        cudaMemcpy(scp_kernel, d_dx_mixed, pad_img_size * sizeof(float), cudaMemcpyDeviceToHost);
        saveMatrixMatlab("cuda_fxmixd", scp_kernel, padw, padh, nc);
        PyErr_Print();

        cudaMemcpy(scp_kernel, d_dy_mixed, pad_img_size * sizeof(float), cudaMemcpyDeviceToHost);
        saveMatrixMatlab("cuda_fymixd", scp_kernel, padw, padh, nc);
        PyErr_Print();

        // TODO: subtract the divergence from upconvolution result (RAVIL)
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);

        alpha = -1.0f * lambda;
        cublasSaxpy(handle, pad_img_size, &alpha, d_div, 1, d_imgUpConv, 1); CUDA_CHECK;

        cudaMemcpy(scp_image, d_imgUpConv, pad_img_size * sizeof(float), cudaMemcpyDeviceToHost);
        saveMatrixMatlab("cuda_gradu", scp_image, padw, padh, nc);
        PyErr_Print();


        // TODO: compute epsilon on GPU
        computeEpsilonGlobalMemCuda(d_epsU, handle, d_imgInPad, d_imgUpConv, pad_img_size, 5e-3);

        // TODO: update output image u = u - eps*grad
            // USE CUBLAS AXPY() FUNCTION HERE
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
        cublasSaxpy(handle, pad_img_size, d_epsU, d_imgUpConv, 1, d_imgInPad, 1);

        cudaMemcpy(scp_image, d_imgInPad, pad_img_size * sizeof(float), cudaMemcpyDeviceToHost);
        saveMatrixMatlab("cuda_upd_image", scp_image, padw, padh, nc);
        PyErr_Print();

        /*debug alpha = -1.0f;*/
        /*debug cublasSaxpy(handle, pad_img_size, &alpha, d_imgUpConv, 1, d_imgInPad, 1);*/


        //$$$$$$$$$$$$$$$$$$ PATCH %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        //convoluton of k^y*y^{t+1}
        /*computeDownConvolutionGlobalMemCuda(d_imgDownConv1, */
                                            /*d_imgInPad, */
                                            /*d_kernel, */
                                            /*padw, */
                                            /*padh, */
                                            /*nc, */
                                            /*mk, nk);*/


        cudaMemcpy(scp_kernel, d_kernel, kn * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(scp_image, d_imgInPad, pad_img_size * sizeof(float), cudaMemcpyDeviceToHost);

        pythonConv2(scp_output, 
                    scp_image, padw, padh, nc,
                    scp_kernel, mk, nk, 1,
                    python_callback, 
                    "valid",
                    "downconv_1");
        PyErr_Print();

        cudaMemcpy(d_imgDownConv1,
                   scp_output, 
                   img_size * sizeof(float), cudaMemcpyHostToDevice);
        //$$$$$$$$$$$$$$$$$$ PATCH %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        //Substraction with f
        alpha = -1.0f;
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
        cublasSaxpy(handle, img_size, &alpha, d_imgIn, 1, d_imgDownConv1, 1); CUDA_CHECK;


        // flip image
        // Ravi has checked that rotation is correct
        for(int c = 0; c < nc; ++c){
            rotateKernel_180(&d_imgPadRot[c * padw * padh],
                             &d_imgInPad[c * padw * padh], 
                             padw,
                             padh); 
        }

        // TODO: perform convolution: k = u * u_pad
        /*computeImageConvolution(d_kernel_temp, mk, nk ,*/
                                /*d_imgDownConv1, d_imgInBuffer, */
                                /*w, h, */
                                /*d_imgPadRot, padw, padh, */
                                /*nc); */


        //$$$$$$$$$$$$$$$$$$ PATCH %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        cudaMemcpy(scp_image, d_imgPadRot, pad_img_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(scp_kernel, d_imgDownConv1, img_size * sizeof(float), cudaMemcpyDeviceToHost);
        
        
        // PYTHON
        pythonConv2(scp_output, 
                    scp_image, padw, padh, nc,
                    scp_kernel, w, h, nc,
                    python_callback, 
                    "valid",
                    "image_image_conv");

        /*std::cout << "C KERNEL" << std::endl;*/
        /*for (int j = 0; j < nk; ++j) {*/
            /*for (int i = 0; i < mk; ++i) {*/
                /*printf("%18.12f\t", scp_output[i + j * mk]);*/
            /*}*/
            /*printf("\n");*/
        /*}*/
        /*std::cout << "C END KERNEL" << std::endl;*/

        cudaMemcpy(d_kernel_temp,
                   scp_output, 
                   kn * sizeof(float), cudaMemcpyHostToDevice);

        /*cv::Mat mDEBUG(padh, padw, mIn.type());*/
        /*convertLayeredToMat(mDEBUG, scp_output); */

        /*showImage("DEBUG", mDEBUG, 150, 200);*/

        //$$$$$$$$$$$$$$$$$$ PATCH %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        /*CHECK whether we passed right kernels (order)*/
        computeEpsilonGlobalMemCuda(d_epsK, handle, d_kernel, d_kernel_temp, kn, 1e-3);

        //update kernel
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
        cublasSaxpy(handle, kn, d_epsK, d_kernel_temp, 1, d_kernel, 1);

        //select non zero kernel
        selectNonZeroGlobalMemCuda(d_kernel, mk, nk);

        //normalise kernel
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
        normaliseGlobalMemCuda(d_kernel, mk, nk);

        cudaMemcpy(scp_output, d_kernel, kn * sizeof(float), cudaMemcpyDeviceToHost);

        saveMatrixMatlab("updated_kernel", scp_output, mk, nk, 1);

        // DEBUG: don't forget uncomment lines bellow
        //update lambda
        /*lambda = 0.99f * lambda;*/
        /*if(lambda < lambda_min){*/
            /*lambda = lambda_min;*/
        /*}*/

    }





	// show input image
    cv::Mat m_dx(padh, padw, mIn.type());
    cv::Mat m_dy(padh, padw, mIn.type());
    cv::Mat m_div(padh, padw, mIn.type());
    cv::Mat mPadImg(padh, padw, mIn.type());
    cv::Mat mImgDownConv0(h, w, mIn.type());
    cv::Mat mImgUpConv(padh, padw, mIn.type());
    cv::Mat mKernel(nk, mk, CV_32FC1);
    
    cudaMemcpy(&epsU, d_epsU, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&epsK, d_epsK, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dx_fw, d_dx_fw, pad_img_size * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
    cudaMemcpy(imgInPad, d_imgInPad, pad_img_size* sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
    cudaMemcpy(dy_fw, d_dy_fw, pad_img_size * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;

    cudaMemcpy(dx_bw, d_dx_bw, pad_img_size * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
    cudaMemcpy(dy_bw, d_dy_bw, pad_img_size * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;

    cudaMemcpy(dx_mixed, d_dx_mixed, pad_img_size * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK;

    cudaMemcpy(dy_mixed, d_dy_mixed, pad_img_size * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK;

    cudaMemcpy(imgDownConv0, d_imgDownConv1, img_size * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK;
    
    cudaMemcpy(kernel, d_kernel, kn * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK;

    cudaMemcpy(div, d_div, pad_img_size * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
    cudaMemcpy(imgUpConv, d_imgUpConv, pad_img_size * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK;



    /*simpleTest(argc, argv);*/
    std::cout << "Value of epsilonU: " << epsU << std::endl;
    std::cout << "Value of epsilonK: " << epsK << std::endl;
    cudaMemcpy(scp_output, d_kernel_temp, kn*sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "----------------------" << std::endl;
    for(int i = 0;  i < nk; ++i){
        for(int j = 0; j < mk; ++j){
            std::cout << scp_output[j + i*mk] << "   ";
        }
        std::cout << std::endl;
    }
   
    std::cout << "----------------------" << std::endl;
    std::cout << "SAXPY" << std::endl;
    for(int i = 0;  i < 5; ++i){
        for(int j = 0; j < 5; ++j){
            std::cout << imgInPad[j + i*padw] << "   ";
        }
        std::cout << std::endl;
    }

    //DEBUG ENDS
    

	// show output image: first convert to interleaved opencv format from the layered raw array
    convertLayeredToMat(m_dx, dx_mixed); 
    convertLayeredToMat(m_dy, dy_mixed); 
    convertLayeredToMat(m_div, div);
    convertLayeredToMat(mPadImg, imgInPad);
    convertLayeredToMat(mImgDownConv0, imgDownConv0);
    convertLayeredToMat(mImgUpConv, imgUpConv);
    convertLayeredToMat(mKernel, kernel);
    
    saveMatrixMatlab("kernel_cross_check", kernel, mk, nk, 1);

    size_t pos_orig_x = 100, pos_orig_y = 50, shift_y = 50; 
    showImage("Input", mIn, pos_orig_x, pos_orig_y);
    /*showImage("dx", m_dx, pos_orig_x + w, pos_orig_y);*/
    /*showImage("dy", m_dy, pos_orig_x, pos_orig_y + w + shift_y);*/
    showImage("divergence", m_div, pos_orig_x + w, pos_orig_y + w + shift_y);
    showImage("Output Image", mPadImg, 100, 140);
    showImage("Kernel", mKernel, 150, 200);
    /*showImage("Down Conv 0", mImgDownConv0, 200, 240);*/
    /*showImage("Up Conv", mImgUpConv, 300, 340);*/

	//convertLayeredToMat(mOut, imgOut);
	//showImage("Output", mOut, 100+w+40, 100);

    // save results
    /*cv::imwrite("image_input.png",mIn*255.f); */
    /*cv::imwrite("image_result.png",m_div*255.f);*/
    /*cv::imwrite("image_kernel.png",mKernel*255.f);*/

    cv::waitKey(0);

    // Free allocated arrays
    delete [] imgIn;
    delete [] imgInPad; 
    delete [] imgOut;

    delete [] dx_fw;
    delete [] dy_fw;
    delete [] dx_bw;
    delete [] dy_bw;
    delete [] dx_mixed;
    delete [] dy_mixed;

    delete [] imgDownConv0;
    delete [] imgUpConv;

    delete [] div;
    delete [] kernel;

    // Python
    delete [] scp_kernel;
    delete [] scp_image;
    delete [] scp_output;

    cudaFree(d_imgIn); CUDA_CHECK;
    cudaFree(d_imgInPad); CUDA_CHECK;
    cudaFree(d_imgPadRot); CUDA_CHECK;
    cudaFree(d_imgInBuffer); CUDA_CHECK;
    cudaFree(d_imgOut); CUDA_CHECK;
    cudaFree(d_kernel); CUDA_CHECK;
    cudaFree(d_kernel_temp); CUDA_CHECK;

    cudaFree(d_dx_fw); CUDA_CHECK;
    cudaFree(d_dy_fw); CUDA_CHECK;
    cudaFree(d_dx_bw); CUDA_CHECK;
    cudaFree(d_dy_bw); CUDA_CHECK;
    cudaFree(d_dx_mixed); CUDA_CHECK;
    cudaFree(d_dy_mixed); CUDA_CHECK;

    cudaFree(d_imgDownConv0); CUDA_CHECK;
    cudaFree(d_imgDownConv1); CUDA_CHECK;
    cudaFree(d_imgUpConv); CUDA_CHECK;
    cudaFree(d_epsU); CUDA_CHECK;
    cudaFree(d_epsK); CUDA_CHECK;

    cudaFree(d_div); CUDA_CHECK;

    cublasDestroy(handle);

    // close all opencv windows
    cv::destroyAllWindows();
    Py_Finalize();
    return 0;
} 




