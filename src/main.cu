// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################

#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "helper.cuh"
#include "divergence.cuh"


int main(int argc,char **argv) {

    // TODO: ADD COMMAND LINE FUNCTIONS LATER

    // parse command line parameters
    const char *params = {
        "{i|image| |input image}"
        "{b|bw|false|load input image as grayscale/black-white}"
        "{m|mk|5|kernel width }"
        "{n|nk|5|kernel height}"
        "{c|cpu|false|compute on CPU}"
       // "{m|mem|0|memory: 0=global, 1=shared, 2=texture}"
    };
    cv::CommandLineParser cmd(argc, argv, params);

    // input image
    std::string inputImage = cmd.get<std::string>("image");
    // number of computation repetitions to get a better run time measurement
    // size_t repeats = (size_t)cmd.get<int>("repeats");
    // load the input image as grayscale
     bool gray = cmd.get<bool>("bw");
	 int mk = cmd.get<int>("mk");
	 int nk = cmd.get<int>("nk");
     bool is_cpu = cmd.get<bool>("cpu");

     std::cout << "mode: " << (is_cpu ? "CPU" : "GPU") << std::endl;


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
	size_t img_size = w * h * nc;
    std::cout << "Image: " << w << " x " << h << std::endl;

    // init kernel
    size_t kn = mk * nk;
	float kernel_init_value = 1.0 / kn;
    float *kernel = new float[kn * sizeof(float)];

    //  initialize kernel to uniform.
	for(int i = 0; i < nc; i++) 
		kernel[i] = kernel_init_value;


    // initialize CUDA context
    // cudaDeviceSynchronize();

    // ### Set the output image format
    cv::Mat mOut(h,w,mIn.type());  // grayscale or color depending on input image, nc layers

    // ### Allocate arrays
    // allocate raw input image array
    float *imgIn = new float[img_size];
    float *imgOut = new float[img_size];
    float *dx = new float[img_size];
    float *dy = new float[img_size];
    float *div = new float[img_size];

    // allocate arrays on GPU
    float *d_imgIn = NULL;
    float *d_imgOut = NULL;
    float *d_dx = NULL;
    float *d_dy = NULL;
    float *d_div = NULL;
    // float *d_kernel = NULL;

    cudaMalloc(&d_imgIn, w * h * nc * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_imgOut , w * h * nc * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_dx , w * h * nc * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_dy , w * h * nc * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_div , w * h * nc * sizeof(float)); CUDA_CHECK;

    // copy input data to GPU 
    cudaMemcpy(d_imgIn, imgIn, w * h * nc * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
    


	// convert range of each channel to [0,1]
	mIn /= 255.0f;
	// init raw input image array (and convert to layered)
	convertMatToLayered(imgIn, mIn);

	// TODO IMPLEMENT THESE FUNCTIONS
	// 1. pre-process: pad image
	// 2. perform blind de-convolution

    computeDivergence(div, dx, dy, imgIn, w, h, nc);

	// padImage(imgIn);
	// computeDeconvolution(imgOut, imgIn, kernel, w, h, nc);


	//cudaMemcpy(imgOut,d_imgOut,nbytes,cudaMemcpyDeviceToHost);

	// show input image
    
    cv::Mat m_dx(h, w, mIn.type());
    cv::Mat m_dy(h, w, mIn.type());
    cv::Mat m_div(h, w, mIn.type());
    
    // copy data from GPU to CPU
    float scale = 10.0;
    for (size_t i = 0; i < (w * h * nc); ++i) {
        dx[i] *= scale;
        dy[i] *= scale;
        div[i] *= scale;
    }
    

	// show output image: first convert to interleaved opencv format from the layered raw array
    convertLayeredToMat(m_dx, dx); 
    convertLayeredToMat(m_dy, dy); 
    convertLayeredToMat(m_div, div);

    size_t pos_orig_x = 100, pos_orig_y = 50, shift_y = 50; 
    showImage("Input", mIn, pos_orig_x, pos_orig_y);
    showImage("dx", m_dx, pos_orig_x + w, pos_orig_y);
    showImage("dy", m_dy, pos_orig_x, pos_orig_y + w + shift_y);
    showImage("divergence", m_div, pos_orig_x + w, pos_orig_y + w + shift_y);

	//convertLayeredToMat(mOut, imgOut);
	//showImage("Output", mOut, 100+w+40, 100);

    // save results
    cv::imwrite("image_input.png",mIn*255.f); 
    /*cv::imwrite("image_result.png",mOut*255.f);*/
    /*cv::imwrite("image_kernel.png",mKernel*255.f);*/

    cv::waitKey(0);

    // Free allocated arrays
    delete [] imgIn;
    delete [] imgOut;
    delete [] dx;
    delete [] dy;
    delete [] div;
    delete [] kernel;

    cudaFree(d_imgIn); CUDA_CHECK;
    cudaFree(d_imgOut); CUDA_CHECK;
    cudaFree(d_dx); CUDA_CHECK;
    cudaFree(d_dy); CUDA_CHECK;
    cudaFree(d_div); CUDA_CHECK;

    // close all opencv windows
    cv::destroyAllWindows();

    return 0;
}



