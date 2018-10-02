// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################

#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "helper.cuh"
#include "downConvolution.cuh"
#include "testdownConv.cuh"
#include "divergence.cuh"


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
        "{lamda|3.5e-4| lamda }"
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
     float lamda = cmd.get<float>("lamda");
     float eps = cmd.get<float>("eps");

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
	size_t img_size = w * h * nc;
    std::cout << "Image: " << w << " x " << h  << " x " << nc << std::endl;

    adjustImageSizeToOdd(mIn, w, h, nc);

    // init kernel
    size_t kn = mk * nk;
	float kernel_init_value = 1.0 / kn;
    float *kernel = new float[kn * sizeof(float)];

    //  initialize kernel to uniform.
	for(int i = 0; i < kn; i++) 
		kernel[i] = kernel_init_value;


    // initialize CUDA context
    // cudaDeviceSynchronize();

    // ### Set the output image format
    cv::Mat mOut(h, w, mIn.type());  // grayscale or color depending on input image, nc layers

    // ### Allocate arrays
    // TODO: CHANGE IMAGE AND IMAGE SIZE TO MAKE ODD IN EACH DIMENSION
    // allocate raw input image array
    float *imgIn = new float[img_size];
    float *imgOut = new float[img_size];
    float *dx_fw = new float[img_size];
    float *dy_fw = new float[img_size];
    float *dx_bw = new float[img_size];
    float *dy_bw = new float[img_size];
    float *dx_mixed = new float[img_size];
    float *dy_mixed = new float[img_size];

    float *div = new float[img_size];

    // TODO: ALLOCATE MEMORY ON GPU
    // allocate arrays on GPU
    float *d_imgIn = NULL;
    float *d_imgOut = NULL;

    float *d_dx_fw = NULL;
    float *d_dy_fw = NULL;
    float *d_dx_bw = NULL;
    float *d_dy_bw = NULL;
    float *d_dx_mixed = NULL;
    float *d_dy_mixed = NULL;

    float *d_div = NULL;
    // float *d_kernel = NULL;

    cudaMalloc(&d_imgIn, w * h * nc * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_imgOut , w * h * nc * sizeof(float)); CUDA_CHECK;

    cudaMalloc(&d_dx_fw, w * h * nc * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_dy_fw, w * h * nc * sizeof(float)); CUDA_CHECK;

    cudaMalloc(&d_dx_bw, w * h * nc * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_dy_bw, w * h * nc * sizeof(float)); CUDA_CHECK;

    cudaMalloc(&d_dx_mixed, w * h * nc * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_dy_mixed, w * h * nc * sizeof(float)); CUDA_CHECK;

    cudaMalloc(&d_div , w * h * nc * sizeof(float)); CUDA_CHECK;

    // copy input data to GPU 
	mIn /= 255.0f;
	convertMatToLayered(imgIn, mIn);
    cudaMemcpy(d_imgIn, imgIn, w * h * nc * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
    


	// convert range of each channel to [0,1]
	// init raw input image array (and convert to layered)

	// TODO:  IMPLEMENT THESE FUNCTIONS
	// 1. pre-process: pad image
	

// 2. perform blind de-convolution

    computeDiffOperatorsCuda(d_div, 
                             d_dx_fw, d_dy_fw,
                             d_dx_bw, d_dy_bw,
                             d_dx_mixed, d_dy_mixed, 
                             d_imgIn, w, h, nc, eps);


	// padImage(imgIn);
	// computeDeconvolution(imgOut, imgIn, kernel, w, h, nc);


	//cudaMemcpy(imgOut,d_imgOut,nbytes,cudaMemcpyDeviceToHost);

	// show input image
    
    cv::Mat m_dx(h, w, mIn.type());
    cv::Mat m_dy(h, w, mIn.type());
    cv::Mat m_div(h, w, mIn.type());
    
    cudaMemcpy(dx_fw, d_dx_fw, w * h * nc * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
    cudaMemcpy(dy_fw, d_dy_fw, w * h * nc * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;

    cudaMemcpy(dx_bw, d_dx_bw, w * h * nc * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
    cudaMemcpy(dy_bw, d_dy_bw, w * h * nc * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;

    cudaMemcpy(dx_mixed, d_dx_mixed, w * h * nc * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK;

    cudaMemcpy(dy_mixed, d_dy_mixed, w * h * nc * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK;

    cudaMemcpy(div, d_div, w * h * nc * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;

    // copy data from GPU to CPU
    float scale = 1.0;
    for (size_t i = 0; i < (w * h * nc); ++i) {
        dx_fw[i] *= scale;
        dy_fw[i] *= scale;

        dx_bw[i] *= scale;
        dy_bw[i] *= scale;

        dx_mixed[i] *= scale;
        dy_mixed[i] *= scale;
        div[i] *= scale;
        /*std::cout << i << "  ---  " << dx_fw[i] << std::endl;*/
    }
    

	// show output image: first convert to interleaved opencv format from the layered raw array
    convertLayeredToMat(m_dx, dx_mixed); 
    convertLayeredToMat(m_dy, dy_mixed); 
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
    cv::imwrite("image_result.png",m_div*255.f);
    /*cv::imwrite("image_kernel.png",mKernel*255.f);*/

    cv::waitKey(0);

    // Free allocated arrays
    delete [] imgIn;
    delete [] imgOut;

    delete [] dx_fw;
    delete [] dy_fw;
    delete [] dx_bw;
    delete [] dx_bw;
    delete [] dy_mixed;
    delete [] dy_mixed;

    delete [] div;
    delete [] kernel;

    cudaFree(d_imgIn); CUDA_CHECK;
    cudaFree(d_imgOut); CUDA_CHECK;

    cudaFree(d_dx_fw); CUDA_CHECK;
    cudaFree(d_dy_fw); CUDA_CHECK;
    cudaFree(d_dx_bw); CUDA_CHECK;
    cudaFree(d_dy_bw); CUDA_CHECK;
    cudaFree(d_dx_mixed); CUDA_CHECK;
    cudaFree(d_dy_mixed); CUDA_CHECK;

    cudaFree(d_div); CUDA_CHECK;

    // close all opencv windows
    cv::destroyAllWindows();

    return 0;
} 




