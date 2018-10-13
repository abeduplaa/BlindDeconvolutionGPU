// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################

#include <iostream>
#include <iomanip>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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
#include "buildPyramid.cuh"
#include "cudnnDownConvolution.cuh"
#include "cudnnUpConvolution.cuh"

#include <Python.h>
#include <numpy/arrayobject.h>

#include "cublas_v2.h"

#define INTERPOLATION_METHOD CV_INTER_CUBIC
/*#define INTERPOLATION_METHOD CV_INTER_LINEAR*/

/*int main(int argc,char **argv)*/
/*{*/
    /*downConvTest();*/
/*}*/


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


 int main(int argc,char **argv) {
    Py_Initialize();
    import_array();
    // TODO: ADD COMMAND LINE FUNCTIONS LATER

    // parse command line parameters
    const char *params = {
        "{image| |input image}"
        "{bw|false|load input image as grayscale/black-white}"
        "{cpu|false|compute on CPU}"
        "{mk|5|kernel width }"
        "{nk|5|kernel height}"
        "{eps|1e-3| epsilon }"
        "{lambda|0.00035|lambda }"
        "{iter|10| iter}"
        "{bc|r| bc}"
    };
    cv::CommandLineParser cmd(argc, argv, params);

    // input image
    std::string inputImage = cmd.get<std::string>("image");

    bool gray = cmd.get<bool>("bw");
	int mk = cmd.get<int>("mk"); 
    mk = (mk <= 0) ? 5 : mk;
	int nk = cmd.get<int>("nk"); nk = (nk <= 0) ? 5 : nk;
    bool is_cpu = cmd.get<bool>("cpu");
    float lambda = cmd.get<float>("lambda"); 
    std::cout << "lambda" << lambda << std::endl;
    lambda = (lambda <= 0) ? 0.00035 : lambda; 
    float lambda_min = 0.00001f;
    float eps = cmd.get<float>("eps"); 
    eps = ( eps <= 0 ) ? 1e-3 : eps;
    int iter = cmd.get<int>("iter"); 
    iter = ( iter <= 0 ) ? 10 : iter;
    std::string bc1 = cmd.get<std::string>("bc");   
	char bc = bc1.c_str()[0];
    BoundaryCondition boundary;
    selectBoundaryCondition(bc, boundary);


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

	size_t img_size = w * h * nc;
    size_t padw = w + mk - 1;
    size_t padh = h + nk - 1;
    size_t pad_img_size = padw * padh * nc;
    std::cout << "Image after Cropping: " << w << " x " << h  << " x " << nc << std::endl;
    std::cout << "Pad Image size" << padw << " x " << padh << " x " << nc << std::endl;

    // init kernel
    size_t kn = mk*nk;
    float *kernel = new float[kn * sizeof(float)];
    //  initialize kernel to uniform.
    initialiseKernel(kernel, mk, nk);

    // initialize CUDA context
    // cudaDeviceSynchronize();

    // Initialize Cublas
    cublasHandle_t handle;
    cublasCreate(&handle);

    //Initialize CUDNN
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    //Pyramid parameter
    int pyramidSize =  0;
    const int smallestKernel = 3;
    const float kernelMultiplier = 1.1f;
    const float lambdaMultiplier = 1.9f;
    const float largestLambda = .11f;
    pyramidSize = pyramidScale(mk,nk, smallestKernel, kernelMultiplier, lambdaMultiplier, lambda, largestLambda);
    std::cout<< "Pyramid Size: " << pyramidSize << std::endl;
    // Build pyramid paramters:
    int *wP = new int[pyramidSize];
    int *hP = new int[pyramidSize];
    int *mP = new int[pyramidSize];
    int *nP = new int[pyramidSize];
    float *lambdas = new float[pyramidSize];

    buildPyramid1(wP, hP, mP, nP, lambdas, w, h, mk, nk, smallestKernel, kernelMultiplier, lambdaMultiplier, lambda, pyramidSize);


	// DEBUGGING:
	std::cout<< "pyramid facts: mP:  ";
	for(int i=0; i<pyramidSize; i++)
	{
		std::cout<< mP[i] << ", ";
	}
	std::cout<< "\n";
    std::cout<< "pyramid facts: nP:  " ;
    for(int i=0; i<pyramidSize; i++)
    {
        std::cout<< nP[i] << ", ";
    }
    std::cout<< "\n";
    std::cout<< "pyramid facts: wP:  " ;
    for(int i=0; i<pyramidSize; i++)
    {
        std::cout<< wP[i] << ", ";
    }
    std::cout<< "\n";
    std::cout<< "pyramid facts: hP:  ";
    for(int i=0; i<pyramidSize; i++)
    {
        std::cout<< hP[i] << ", ";
    }
    std::cout<< "\n";
    std::cout<< "pyramid facts: lambdas:  ";
    for(int i=0; i<pyramidSize; i++)
    {
        std::cout<< lambdas[i] << ", ";
    }
    std::cout<< "\n";


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


    // TODO: ALLOCATE MEMORY ON GPU
    // allocate arrays on GPU
    float *d_imgIn = NULL;
    float *d_imgInPad = NULL;
    float *d_imgInPadTemp = NULL;
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

    float *d_imgDownConv1Rot = NULL;

    float *d_imgUpConv = NULL;

    float *d_div = NULL;
    float *d_kernel = NULL;
    float *d_kernel_temp = NULL;

    float *d_epsU = NULL;
    float *d_epsK = NULL;

    float *d_comm_buffer = NULL;

    cudaMalloc(&d_imgIn, img_size * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_imgInBuffer, img_size * sizeof(float)); CUDA_CHECK;

    // TODO: be sure the size id right (RAVI)
    cudaMalloc(&d_imgInPad, pad_img_size * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_imgInPadTemp, pad_img_size * sizeof(float)); CUDA_CHECK;
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

    cudaMalloc(&d_imgDownConv1Rot, img_size * sizeof(float)); CUDA_CHECK;

    cudaMalloc(&d_imgUpConv, pad_img_size * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_epsU, sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_epsK, sizeof(float)); CUDA_CHECK;

    cudaMalloc(&d_comm_buffer, pad_img_size * sizeof(float)); CUDA_CHECK;

	mIn /= 255.0f;
	convertMatToLayered(imgIn, mIn);
    cudaMemcpy(d_imgIn, imgIn, img_size * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
    cudaMemcpy(d_kernel, kernel, kn * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;

    padImgGlobalMemCuda(d_imgInPad, 
                        d_imgIn, 
                        w, h, nc, 
                        mk, nk, 
                        boundary);
    //Start loop  for pyramid
    
    int START_LEVEL = pyramidSize - 1;
    /*int START_LEVEL = 0;*/
    std::cout<< "Starting level" << START_LEVEL << std::endl;
    for(int level = START_LEVEL; level >= 0; --level){
		//break;
        //copy image and padded image back to CPU
        std::cout << "Level Number:   " << level << 
            ", lambda: " << lambdas[level] << 
            "mk" << mP[level] << "nk" << nP[level]  <<std::endl;
        cudaMemcpy(imgIn, d_imgIn, img_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(imgInPad, d_imgInPad, pad_img_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(kernel, d_kernel, kn * sizeof(float), cudaMemcpyDeviceToHost);
        cv::Mat mImgOld(h, w, mIn.type());
        cv::Mat mImgPadOld(padh, padw, mIn.type());
        cv::Mat mKernelOld(nk, mk, CV_32F);
        //update the dimensions
        w = wP[level]; h = hP[level];
        mk = mP[level]; nk = nP[level];
        padw = w + mk - 1;
        padh = h + nk - 1;
        img_size = w * h * nc;
        pad_img_size = padw * padh * nc;
        kn = mk * nk;
        lambda = lambdas[level];
        //convert array to image
        cv::Mat mImgResize(h, w, mIn.type());
        cv::Mat mImgPadResize(padh, padw, mIn.type());
        cv::Mat mKernelResize(nk, mk, CV_32F);
        convertLayeredToMat(mImgOld, imgIn);
        convertLayeredToMat(mImgPadOld, imgInPad);
        convertLayeredToMat(mKernelOld, kernel);
        //Display image of previous iteration
        //convert Layered to mat
        //resize image
        cv::resize(mImgOld, mImgResize, mImgResize.size(), 0, 0, INTERPOLATION_METHOD);
        cv::resize(mImgPadOld, mImgPadResize, mImgPadResize.size(), 0, 0, INTERPOLATION_METHOD);
        cv::resize(mKernelOld, mKernelResize, mKernelResize.size(), 0, 0, INTERPOLATION_METHOD);
        //Copy image to array
        /*showImage("Image resized", mImgResize, 200, 200);*/
        /*showImage("Padded resized", mImgPadResize, 100, 200);*/
        /*cv::Mat mKernelOut(nk, mk, CV_8UC1);*/
        /*mKernelResize.convertTo(mKernelOut, CV_8UC1, 255.0 * (mk*nk), 0); */
        /*cv::namedWindow("Kernel", CV_WINDOW_NORMAL);*/
        /*cv::imshow("Kernel", mKernelOut);*/
        std::string img_name = "Image_" + std::to_string(level);
        std::string kernel_name = "Kernel_" + std::to_string(level);
        
        //copy image to GPU
        convertMatToLayered(imgIn, mImgResize);
        convertMatToLayered(imgInPad, mImgPadResize);
        convertMatToLayered(kernel, mKernelResize);
        saveMatrixMatlab(img_name.c_str(),
                        imgInPad,
                        padw,
                        padh,
                        nc);
        PyErr_Print();
        saveMatrixMatlab(kernel_name.c_str(),
                        kernel,
                        mk,
                        nk,
                        1);
        PyErr_Print();
        cudaMemcpy(d_kernel, kernel, kn * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_imgIn, imgIn, img_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_imgInPad, imgInPad, pad_img_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_imgInPadTemp, imgInPad, pad_img_size * sizeof(float), cudaMemcpyHostToDevice);

        //select non zero kernel
        selectNonZeroGlobalMemCuda(d_kernel, mk, nk);
        /*cudaThreadSynchronize();*/

        //normalise kernel
        normaliseGlobalMemCuda(d_kernel, mk, nk);
        /*cudaThreadSynchronize();*/

        /*break;*/
    //}
    

    //Declaration of CUDNN descriptors
    cudnnTensorDescriptor_t input_descriptor1, input_descriptor2, input_descriptor3;
    cudnnFilterDescriptor_t kernel_descriptor1, kernel_descriptor2, kernel_descriptor3;
    cudnnConvolutionDescriptor_t convolution_descriptor1, convolution_descriptor2, convolution_descriptor3;
    cudnnTensorDescriptor_t output_descriptor1, output_descriptor2, output_descriptor3;
    cudnnConvolutionFwdAlgo_t convolution_algorithm1, convolution_algorithm2, convolution_algorithm3;

    //workspace for cudnn
    void* d_workspace1{nullptr};
    void* d_workspace2{nullptr};
    void* d_workspace3{nullptr};

    //Defining Descriptors
    size_t workspace_bytes1 = createDescriptorsdc0(input_descriptor1,
                                               kernel_descriptor1,
                                               convolution_descriptor1,
                                               output_descriptor1,
                                               convolution_algorithm1,
                                               cudnn,
                                               padw,
                                               padh,
                                               mk,
                                               nk,
                                               nc);
    cudaMalloc(&d_workspace1, workspace_bytes1);

    size_t workspace_bytes2 = createDescriptorsUp(input_descriptor2,
                                               kernel_descriptor2,
                                               convolution_descriptor2,
                                               output_descriptor2,
                                               convolution_algorithm2,
                                               cudnn,
                                               w,
                                               h,
                                               mk,
                                               nk,
                                               nc);
    cudaMalloc(&d_workspace2, workspace_bytes2);

    size_t workspace_bytes3 = createDescriptorsdc1(input_descriptor3,
                                               kernel_descriptor3,
                                               convolution_descriptor3,
                                               output_descriptor3,
                                               convolution_algorithm3,
                                               cudnn,
                                               padw,
                                               padh,
                                               w,
                                               h,
                                               nc);
    cudaMalloc(&d_workspace3, workspace_bytes3);
   
    // Blind Deconvolution algorithm starts here

    for(int iterations = 0; iterations < iter; ++iterations){
        //if(iterations % 100 == 0)
            std::cout << "Iteration num:  " << iterations << std::endl;

        // TODO: compute(mirror, rotate) kernel
        rotateKernel_180(d_kernel_temp, d_kernel, mk, nk); 
        /*cudaThreadSynchronize();*/
        //Our code
        /*computeDownConvolutionGlobalMemCuda(d_imgDownConv0, */
                                            /*d_imgInPadTemp, */
                                            /*d_kernel_temp, */
                                            /*padw, */
                                            /*padh, */
                                            /*nc, */
                                            /*mk, */
                                            /*nk);*/
        //CUDNN
        callConvolutiondc0(d_imgDownConv0,
                         d_imgInPadTemp,
                         d_kernel_temp,
                         padw, padh,
                         mk, nk,
                         w,h,nc,
                         input_descriptor1,
                         kernel_descriptor1,
                         convolution_descriptor1,
                         output_descriptor1,
                         convolution_algorithm1,
                         d_workspace1,
                         cudnn,
                         workspace_bytes1);
        /*cudaMemcpy(imgUpConv, d_imgDownConv0, img_size * sizeof(float), cudaMemcpyDeviceToHost);*/
    /*saveMatrixMatlab("cuda_conv1",*/
                    /*imgUpConv,*/
                    /*w,*/
                    /*h,*/
                    /*nc);*/
    /*PyErr_Print();*/
        /*cudaThreadSynchronize();*/
        /*if(iterations == 1)*/
            /*break;*/
        // DONE: cublas subtract k(+)*u - f. Move that to a separate function
        alpha = -1.0f;
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
        cublasSaxpy(handle, img_size, &alpha, d_imgIn, 1, d_imgDownConv0, 1); CUDA_CHECK;
        /*cudaThreadSynchronize();*/
        
        //Our Call
        /*computeUpConvolutionGlobalMemCuda(d_imgUpConv, d_imgDownConv0, d_kernel, w, h, nc, mk, nk);*/

        //CUDNN
        callConvolutionUp(d_imgUpConv,
                         d_imgDownConv0,
                         d_kernel,
                         w, h,
                         mk, nk,
                         padw, padh, nc,
                         input_descriptor2,
                         kernel_descriptor2,
                         convolution_descriptor2,
                         output_descriptor2,
                         convolution_algorithm2,
                         d_workspace2,
                         cudnn,
                         workspace_bytes2);

        /*cudaMemcpy(imgUpConv, d_imgUpConv, pad_img_size * sizeof(float), cudaMemcpyDeviceToHost);*/
    /*saveMatrixMatlab("cuda_conv2",*/
                    /*imgUpConv,*/
                    /*padw,*/
                    /*padh,*/
                    /*nc);*/
    /*PyErr_Print();*/
        /*cudaThreadSynchronize();*/

        // compute gradient and divergence
        computeDiffOperatorsCuda(d_div, 
                                 d_dx_fw, d_dy_fw,
                                 d_dx_bw, d_dy_bw,
                                 d_dx_mixed, d_dy_mixed, 
                                 d_imgInPadTemp, padw, padh, nc, eps);
        
        /*cudaMemcpy(imgUpConv, d_div, pad_img_size * sizeof(float), cudaMemcpyDeviceToHost);*/
    /*saveMatrixMatlab("cuda_div",*/
                    /*imgUpConv,*/
                    /*padw,*/
                    /*padh,*/
                    /*nc);*/
    /*PyErr_Print();*/
        /*cudaThreadSynchronize();*/
        

        // TODO: subtract the divergence from upconvolution result (RAVIL)
        alpha = -1.0f * lambda;
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
        cublasSaxpy(handle, pad_img_size, &alpha, d_div, 1, d_imgUpConv, 1); CUDA_CHECK;
        /*cudaThreadSynchronize();*/
        /*cudaMemcpy(imgUpConv, d_imgUpConv, pad_img_size * sizeof(float), cudaMemcpyDeviceToHost);*/
    /*saveMatrixMatlab("cuda_subU",*/
                    /*imgUpConv,*/
                    /*padw,*/
                    /*padh,*/
                    /*nc);*/
    /*PyErr_Print();*/
        // TODO: compute epsilon on GPU
        computeEpsilonGlobalMemCuda(d_epsU, handle, d_imgInPadTemp, d_imgUpConv, pad_img_size, 5e-3);
        /*cudaThreadSynchronize();*/
        

        // TODO: update output image u = u - eps*grad
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
        cublasSaxpy(handle, pad_img_size, d_epsU, d_imgUpConv, 1, d_imgInPadTemp, 1);
        /*cudaThreadSynchronize();*/

        cudaMemcpy(&epsU, d_epsU, sizeof(float), cudaMemcpyDeviceToHost);
        /*cudaMemcpy(imgUpConv, d_imgInPadTemp, pad_img_size * sizeof(float), cudaMemcpyDeviceToHost);*/
    /*saveMatrixMatlab("cuda_updateU",*/
                    /*imgUpConv,*/
                    /*padw,*/
                    /*padh,*/
                    /*nc);*/
    /*PyErr_Print();*/

        //convoluton of k^y*y^{t+1}
        //Our implementation
        /*computeDownConvolutionGlobalMemCuda(d_imgDownConv1, */
                                            /*d_imgInPadTemp, */
                                            /*d_kernel_temp, */
                                            /*padw, */
                                            /*padh, */
                                            /*nc, */
                                            /*mk, nk);*/
        //CUDNN
        callConvolutiondc0(d_imgDownConv1,
                         d_imgInPadTemp,
                         d_kernel_temp,
                         padw, padh,
                         mk, nk,
                         w,h,nc,
                         input_descriptor1,
                         kernel_descriptor1,
                         convolution_descriptor1,
                         output_descriptor1,
                         convolution_algorithm1,
                         d_workspace1,
                         cudnn,
                         workspace_bytes1);
        cudaMemcpy(imgUpConv, d_imgDownConv1, img_size * sizeof(float), cudaMemcpyDeviceToHost);
    /*saveMatrixMatlab("cuda_conv3",*/
                    /*imgUpConv,*/
                    /*w,*/
                    /*h,*/
                    /*nc);*/
    /*PyErr_Print();*/
        /*cudaThreadSynchronize();*/

        //Substraction with f
        alpha = -1.0f;
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
        cublasSaxpy(handle, img_size, &alpha, d_imgIn, 1, d_imgDownConv1, 1); CUDA_CHECK;
        /*cudaThreadSynchronize();*/

        // flip image
        // Ravi has checked that rotation is correct
        for(int c = 0; c < nc; ++c){
            rotateKernel_180(&d_imgPadRot[c*padw*padh], &d_imgInPadTemp[c*padw*padh], 
                    padw, padh);
            /*cudaThreadSynchronize(); */
            rotateKernel_180(&d_imgDownConv1Rot[c*w*h], &d_imgDownConv1[c*w*h], w, h);
            /*cudaThreadSynchronize();*/
        }

        // TODO: perform convolution: k = u * u_pad
        //Our implementation
        computeImageConvolution(d_kernel_temp, mk, nk ,
                                d_imgDownConv1Rot, d_imgInBuffer, 
                                w, h, 
                                d_imgPadRot, padw, padh, 
                                nc); 
        //CUDNN
        /*callConvolutiondc1(d_kernel_temp,*/
                         /*d_imgPadRot,*/
                         /*d_imgDownConv1Rot,*/
                         /*input_descriptor3,*/
                         /*kernel_descriptor3,*/
                         /*convolution_descriptor3,*/
                         /*output_descriptor3,*/
                         /*convolution_algorithm3,*/
                         /*d_workspace3,*/
                         /*cudnn,*/
                         /*workspace_bytes3);*/

        /*cudaThreadSynchronize();*/
        
        /*cudaMemcpy(imgUpConv, d_kernel_temp, kn * sizeof(float), cudaMemcpyDeviceToHost);*/
    /*saveMatrixMatlab("cuda_conv4",*/
                    /*imgUpConv,*/
                    /*mk,*/
                    /*nk,*/
                    /*1);*/
    /*PyErr_Print();*/
        computeEpsilonGlobalMemCuda(d_epsK, handle, d_kernel, d_kernel_temp, kn, 1e-3);
        /*cudaThreadSynchronize();*/
        /*cudaMemcpy(kernel, d_kernel_temp, kn*sizeof(float), cudaMemcpyDeviceToHost);*/

        /*std::cout << "Grad k" << std::endl;*/
        /*for(int i = 0;  i < nk; ++i){*/
            /*for(int j = 0; j < mk; ++j){*/
                /*std::cout << kernel[j + i*mk] << "   ";*/
            /*}*/
            /*std::cout << std::endl;*/
        /*}*/

        //update kernel
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
        cublasSaxpy(handle, kn, d_epsK, d_kernel_temp, 1, d_kernel, 1);
        /*cudaThreadSynchronize();*/

        /*cudaMemcpy(kernel, d_kernel, kn*sizeof(float), cudaMemcpyDeviceToHost);*/

        /*std::cout << "After time update" << std::endl;*/
        /*for(int i = 0;  i < nk; ++i){*/
            /*for(int j = 0; j < mk; ++j){*/
                /*std::cout << kernel[j + i*mk] << "   ";*/
            /*}*/
            /*std::cout << std::endl;*/
        /*}*/

        //select non zero kernel
        selectNonZeroGlobalMemCuda(d_kernel, mk, nk);
        /*cudaThreadSynchronize();*/

        //normalise kernel
        normaliseGlobalMemCuda(d_kernel, mk, nk);
        /*cudaThreadSynchronize();*/

        //update lambda
        /*lambda = 0.99f * lambda;*/
        /*if(lambda < lambda_min){*/
            /*lambda = lambda_min;*/
        /*}*/

    
    }
    //Dec
    
    for(int iterations = 0; iterations < iter; ++iterations){
        //if(iterations % 100 == 0)
            std::cout << "Dec Iteration num:  " << iterations << std::endl;

        // TODO: compute(mirror, rotate) kernel
        rotateKernel_180(d_kernel_temp, d_kernel, mk, nk); 
        /*cudaThreadSynchronize();*/
        //Our code
        /*computeDownConvolutionGlobalMemCuda(d_imgDownConv0, */
                                            /*d_imgInPad, */
                                            /*d_kernel_temp, */
                                            /*padw, */
                                            /*padh, */
                                            /*nc, */
                                            /*mk, */
                                            /*nk);*/
        //CUDNN
        callConvolutiondc0(d_imgDownConv0,
                         d_imgInPad,
                         d_kernel_temp,
                         padw, padh,
                         mk, nk,
                         w,h,nc,
                         input_descriptor1,
                         kernel_descriptor1,
                         convolution_descriptor1,
                         output_descriptor1,
                         convolution_algorithm1,
                         d_workspace1,
                         cudnn,
                         workspace_bytes1);
        /*cudaThreadSynchronize();*/
        /*if(iterations == 1)*/
            /*break;*/
        // DONE: cublas subtract k(+)*u - f. Move that to a separate function
        alpha = -1.0f;
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
        cublasSaxpy(handle, img_size, &alpha, d_imgIn, 1, d_imgDownConv0, 1); CUDA_CHECK;
        /*cudaThreadSynchronize();*/
        
        //Our Call
        /*computeUpConvolutionGlobalMemCuda(d_imgUpConv, d_imgDownConv0, d_kernel, w, h, nc, mk, nk);*/

        //CUDNN
        callConvolutionUp(d_imgUpConv,
                         d_imgDownConv0,
                         d_kernel,
                         w, h,
                         mk, nk,
                         padw, padh, nc,
                         input_descriptor2,
                         kernel_descriptor2,
                         convolution_descriptor2,
                         output_descriptor2,
                         convolution_algorithm2,
                         d_workspace2,
                         cudnn,
                         workspace_bytes2);

        /*cudaThreadSynchronize();*/

        // compute gradient and divergence
        computeDiffOperatorsCuda(d_div, 
                                 d_dx_fw, d_dy_fw,
                                 d_dx_bw, d_dy_bw,
                                 d_dx_mixed, d_dy_mixed, 
                                 d_imgInPad, padw, padh, nc, eps);
        /*cudaThreadSynchronize();*/
        

        // TODO: subtract the divergence from upconvolution result (RAVIL)
        alpha = -1.0f * lambda;

        cublasSaxpy(handle, pad_img_size, &alpha, d_div, 1, d_imgUpConv, 1); CUDA_CHECK;
        // TODO: compute epsilon on GPU
        computeEpsilonGlobalMemCuda(d_epsU, handle, d_imgInPad, d_imgUpConv, pad_img_size, 5e-3);
        /*cudaThreadSynchronize();*/
        

        // TODO: update output image u = u - eps*grad
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
        cublasSaxpy(handle, pad_img_size, d_epsU, d_imgUpConv, 1, d_imgInPad, 1);
        /*cudaThreadSynchronize();*/

    }
    destroyDescriptors(input_descriptor1,
                       kernel_descriptor1,
                       convolution_descriptor1,
                       output_descriptor1);
    cudaFree(d_workspace1);
    destroyDescriptors(input_descriptor2,
                       kernel_descriptor2,
                       convolution_descriptor2,
                       output_descriptor2);
    cudaFree(d_workspace2);
    destroyDescriptors(input_descriptor3,
                       kernel_descriptor3,
                       convolution_descriptor3,
                       output_descriptor3);
    cudaFree(d_workspace3);
   }
	// show input image
    cv::Mat m_dx(padh, padw, mIn.type());
    cv::Mat m_dy(padh, padw, mIn.type());
    cv::Mat m_div(padh, padw, mIn.type());
    cv::Mat mPadImg(padh, padw, mIn.type());
    cv::Mat mImgDownConv0(h, w, mIn.type());
    cv::Mat mImgUpConv(padh, padw, mIn.type());
    cv::Mat mKernel(nk, mk, CV_32F);
    
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

    cudaMemcpy(imgDownConv0, d_imgDownConv0, img_size * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK;
    
    cudaMemcpy(kernel, d_kernel, kn * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK;

    cudaMemcpy(div, d_div, pad_img_size * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
    cudaMemcpy(imgUpConv, d_imgUpConv, pad_img_size * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK;

    //copy data from GPU to CPU
    /*float scale = 10.0;*/
    /*for (size_t i = 0; i < pad_img_size; ++i) {*/
        /*dx_fw[i] *= scale;*/
        /*dy_fw[i] *= scale;*/

        /*dx_bw[i] *= scale;*/
        /*dy_bw[i] *= scale;*/

        /*dx_mixed[i] *= scale;*/
        /*dy_mixed[i] *= scale;*/
        /*div[i] *= scale;*/
        /*std::cout << i << "  ---  " << dx_fw[i] << std::endl;*/
    /*}*/
    //DEBUG STARTS
    /*simpleTest(argc, argv);*/
    std::cout << "Value of epsilonU: " << epsU << std::endl;
    std::cout << "Value of epsilonK: " << epsK << std::endl;
    std::cout << "Value of lambda at the end: " << lambda << std::endl;

    /*for(int i = 0;  i < nk; ++i){*/
        /*for(int j = 0; j < mk; ++j){*/
            /*std::cout << std::setprecision(7) << kernel[j + i*mk] << "   ";*/
        /*}*/
        /*std::cout << std::endl;*/
    /*}*/
    /*std::cout << "YOLO" << std::endl; */
    /*for(int i = 0;  i < 5; ++i){*/
        /*for(int j = 0; j < 5; ++j){*/
            /*std::cout << std::setprecision(7) << imgDownConv0[j +  i*w] << "   ";*/
        /*}*/
        /*std::cout << std::endl;*/
    /*}*/

    //DEBUG ENDS
    

	// show output image: first convert to interleaved opencv format from the layered raw array
    convertLayeredToMat(m_dx, dx_mixed); 
    convertLayeredToMat(m_dy, dy_mixed); 
    convertLayeredToMat(m_div, div);
    convertLayeredToMat(mPadImg, imgInPad);
    convertLayeredToMat(mImgDownConv0, imgDownConv0);
    convertLayeredToMat(mImgUpConv, imgUpConv);
    convertLayeredToMat(mKernel, kernel);

    
    /*saveMatrixMatlab("interp_image",*/
                    /*imgIn,*/
                    /*w,*/
                    /*h,*/
                    /*nc);*/
    /*PyErr_Print();*/
    saveMatrixMatlab("output_image",
                    imgInPad,
                    padw,
                    padh,
                    nc);
    PyErr_Print();
    saveMatrixMatlab("output_kernel",
                    kernel,
                    mk,
                    nk,
                    1);
    PyErr_Print();

    size_t pos_orig_x = 100, pos_orig_y = 50, shift_y = 50; 
    showImage("Input", mIn, pos_orig_x, pos_orig_y);
    /*showImage("dx", m_dx, pos_orig_x + w, pos_orig_y);*/
    /*showImage("dy", m_dy, pos_orig_x, pos_orig_y + w + shift_y);*/
    showImage("divergence", m_div, pos_orig_x + w, pos_orig_y + w + shift_y);
    showImage("Output Image", mPadImg, 100, 140);
    /*showImage("Kernel", mKernel*10000, 150, 200);*/
    cv::Mat mKernelOut(nk, mk, CV_8UC1);
    mKernel.convertTo(mKernelOut, CV_8UC1, 255.0 * (mk*nk), 0); 
    /*cv::applyColorMap(mKernelOut, mKernelOut, cv::COLORMAP_JET);*/
    cv::namedWindow("Kernel", CV_WINDOW_NORMAL);
    cv::imshow("Kernel", mKernelOut);
    /*showImage("Down Conv 0", mImgDownConv0, 200, 240);*/
    /*showImage("Up Conv", mImgUpConv, 300, 340);*/

	//convertLayeredToMat(mOut, imgOut);
	//showImage("Output", mOut, 100+w+40, 100);

    // save results
    /*cv::imwrite("image_input.png",mIn*255.f); */
    /*cv::imwrite("image_result.png",m_div*255.f);*/
    /*cv::imwrite("image_kernel.png",mKernel*255.f);*/
    /*cv::imwrite("out.png",mPadImg);*/

    cv::FileStorage file("Kernel.txt", cv::FileStorage::WRITE);
    file << "Kernel" << mKernel;
    cv::FileStorage file1("Output.txt", cv::FileStorage::WRITE);
    file1 << "Output" << mPadImg;

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

    cudaFree(d_imgIn); CUDA_CHECK;
    cudaFree(d_imgInPad); CUDA_CHECK;
    cudaFree(d_imgInPadTemp); CUDA_CHECK;
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
    cudaFree(d_imgDownConv1Rot); CUDA_CHECK;
    cudaFree(d_imgUpConv); CUDA_CHECK;
    cudaFree(d_epsU); CUDA_CHECK;
    cudaFree(d_epsK); CUDA_CHECK;

    cudaFree(d_div); CUDA_CHECK;

    cublasDestroy(handle);
    cudnnDestroy(cudnn);

    // close all opencv windows
    cv::destroyAllWindows();
    Py_Finalize();

    return 0;
} 




