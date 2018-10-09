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

#include "cublas_v2.h"


/*int main(int argc,char **argv)*/
/*{*/
    /*downConvTest();*/
/*}*/



 int main(int argc,char **argv) {

    // TODO: ADD COMMAND LINE FUNCTIONS LATER

    // parse command line parameters
    const char *params = {
        "{image| |input image}"
        "{bw|false|load input image as grayscale/black-white}"
        "{mk|5|kernel width }"
        "{nk|5|kernel height}"
        "{cpu|false|compute on CPU}"
        "{eps|1e-3| epsilon }"
        "{lambda|0.0068| lambda }"
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
     float lambda = cmd.get<float>("lambda"); lambda = (lambda <= 0) ? 0.0068 : lambda; 
     float lambda_min = 0.0006f;
     float eps = cmd.get<float>("eps"); eps = ( eps <= 0 ) ? 1e-3 : eps;
     int iter = cmd.get<int>("iter"); iter = ( iter <= 0 ) ? 1 : iter;


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

    cudaMalloc(&d_imgDownConv1Rot, img_size * sizeof(float)); CUDA_CHECK;

    cudaMalloc(&d_imgUpConv, pad_img_size * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_epsU, sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_epsK, sizeof(float)); CUDA_CHECK;

	mIn /= 255.0f;
	convertMatToLayered(imgIn, mIn);
    cudaMemcpy(d_imgIn, imgIn, img_size * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
    cudaMemcpy(d_kernel, kernel, kn * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;

    padImgGlobalMemCuda(d_imgInPad, d_imgIn, w, h, nc, mk, nk);

    for(int iterations = 0; iterations < iter; ++iterations){
        std::cout << "Iteration num:  " << iterations << std::endl;

        // TODO: compute(mirror, rotate) kernel
        rotateKernel_180(d_kernel_temp, d_kernel, mk, nk); 
        cudaThreadSynchronize();

        computeDownConvolutionGlobalMemCuda(d_imgDownConv0, 
                                            d_imgInPad, 
                                            d_kernel_temp, 
                                            padw, 
                                            padh, 
                                            nc, 
                                            mk, 
                                            nk);
        cudaThreadSynchronize();
        /*if(iterations == 1)*/
            /*break;*/
        // DONE: cublas subtract k(+)*u - f. Move that to a separate function
        alpha = -1.0f;
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
        cublasSaxpy(handle, img_size, &alpha, d_imgIn, 1, d_imgDownConv0, 1); CUDA_CHECK;
        cudaThreadSynchronize();

        // TODO: check the list of  parameters 
        computeUpConvolutionGlobalMemCuda(d_imgUpConv, d_imgDownConv0, d_kernel, w, h, nc, mk, nk);
        cudaThreadSynchronize();

        // compute gradient and divergence
        computeDiffOperatorsCuda(d_div, 
                                 d_dx_fw, d_dy_fw,
                                 d_dx_bw, d_dy_bw,
                                 d_dx_mixed, d_dy_mixed, 
                                 d_imgInPad, padw, padh, nc, 1.0f, eps);
        cudaThreadSynchronize();
        

        // TODO: subtract the divergence from upconvolution result (RAVIL)
        alpha = -1.0f * lambda;

        cublasSaxpy(handle, pad_img_size, &alpha, d_div, 1, d_imgUpConv, 1); CUDA_CHECK;
        // TODO: compute epsilon on GPU
        computeEpsilonGlobalMemCuda(d_epsU, handle, d_imgInPad, d_imgUpConv, pad_img_size, 5e-3);
        cudaThreadSynchronize();
        

        // TODO: update output image u = u - eps*grad
            // USE CUBLAS AXPY() FUNCTION HERE
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
        cublasSaxpy(handle, pad_img_size, d_epsU, d_imgUpConv, 1, d_imgInPad, 1);
        cudaThreadSynchronize();


        //convoluton of k^y*y^{t+1}
        computeDownConvolutionGlobalMemCuda(d_imgDownConv1, 
                                            d_imgInPad, 
                                            d_kernel_temp, 
                                            padw, 
                                            padh, 
                                            nc, 
                                            mk, nk);
        cudaThreadSynchronize();

        //Substraction with f
        alpha = -1.0f;
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
        cublasSaxpy(handle, img_size, &alpha, d_imgIn, 1, d_imgDownConv1, 1); CUDA_CHECK;
        cudaThreadSynchronize();

        // flip image
        // Ravi has checked that rotation is correct
        for(int c = 0; c < nc; ++c){
            rotateKernel_180(&d_imgPadRot[c*padw*padh], &d_imgInPad[c*padw*padh], 
                    padw, padh); 
            rotateKernel_180(&d_imgDownConv1Rot[c*w*h], &d_imgDownConv1[c*w*h], w, h);
            cudaThreadSynchronize();
        }

        // TODO: perform convolution: k = u * u_pad
        // computeDownConvolutionGlobalMemCuda1(d_kernel_temp, 
        //                                    d_imgPadRot, 
        //                                    d_imgDownConv1Rot, 
        //                                    padw, 
        //                                    padh, 
        //                                    nc, 
        //                                    w, h);

        computeImageConvolution(d_kernel_temp, mk, nk ,
                                d_imgDownConv1Rot, d_imgInBuffer, 
                                w, h, 
                                d_imgPadRot, padw, padh, 
                                nc); 
        cudaThreadSynchronize();
        
        computeEpsilonGlobalMemCuda(d_epsK, handle, d_kernel, d_kernel_temp, kn, 1e-3);
        cudaThreadSynchronize();
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
        cudaThreadSynchronize();

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
        cudaThreadSynchronize();

        //normalise kernel
        normaliseGlobalMemCuda(d_kernel, mk, nk);
        cudaThreadSynchronize();

        //update lambda
        lambda = 0.99f * lambda;
        if(lambda < lambda_min){
            lambda = lambda_min;
        }

    }
	// convert range of each channel to [0,1]
	// init raw input image array (and convert to layered)

	// TODO:  IMPLEMENT THESE FUNCTIONS

	// 1. pre-process: pad image
	

// 2. perform blind de-convolution



	// padImage(imgIn);
	// computeDeconvolution(imgOut, imgIn, kernel, w, h, nc);


	//cudaMemcpy(imgOut,d_imgOut,nbytes,cudaMemcpyDeviceToHost);

	// show input image
    
    cv::Mat m_dx(padh, padw, mIn.type());
    cv::Mat m_dy(padh, padw, mIn.type());
    cv::Mat m_div(padh, padw, mIn.type());
    cv::Mat mPadImg(padh, padw, mIn.type());
    cv::Mat mImgDownConv0(h, w, mIn.type());
    cv::Mat mImgUpConv(padh, padw, mIn.type());
    cv::Mat mKernel(nk, mk, 1);
    
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

    for(int i = 0;  i < nk; ++i){
        for(int j = 0; j < mk; ++j){
            std::cout << std::setprecision(7) << kernel[j + i*mk] << "   ";
        }
        std::cout << std::endl;
    }
    std::cout << "YOLO" << std::endl; 
    for(int i = 0;  i < 5; ++i){
        for(int j = 0; j < 5; ++j){
            std::cout << std::setprecision(7) << imgDownConv0[j +  i*w] << "   ";
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
    cudaFree(d_imgDownConv1Rot); CUDA_CHECK;
    cudaFree(d_imgUpConv); CUDA_CHECK;
    cudaFree(d_epsU); CUDA_CHECK;
    cudaFree(d_epsK); CUDA_CHECK;

    cudaFree(d_div); CUDA_CHECK;

    cublasDestroy(handle);

    // close all opencv windows
    cv::destroyAllWindows();

    return 0;
} 




