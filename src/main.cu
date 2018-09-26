// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################

#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main(int argc,char **argv)
{

// TODO: ADD COMMAND LINE FUNCTIONS LATER


    // parse command line parameters
    const char *params = {
        "{i|image| |input image}"
       // "{b|bw|false|load input image as grayscale/black-white}"
       // "{s|sigma|3.0|sigma}"
       // "{r|repeats|1|number of computation repetitions}"
       // "{c|cpu|false|compute on CPU}"
       // "{m|mem|0|memory: 0=global, 1=shared, 2=texture}"
    };
    cv::CommandLineParser cmd(argc, argv, params);

    // input image
    std::string inputImage = cmd.get<std::string>("image");
    // number of computation repetitions to get a better run time measurement
    // size_t repeats = (size_t)cmd.get<int>("repeats");
    // load the input image as grayscale
    // bool gray = cmd.get<bool>("bw");
    // compute on CPU
    // bool cpu = cmd.get<bool>("cpu");
    // std::cout << "mode: " << (cpu ? "CPU" : "GPU") << std::endl;


    // read input frame
    cv::Mat mIn;
        // load the input image using opencv (load as grayscale if "gray==true", otherwise as is (may be color or grayscale))
        mIn = cv::imread(inputImage.c_str(), (gray ? CV_LOAD_IMAGE_GRAYSCALE : -1));
    }
    // check
    if (mIn.empty())
    {
        std::cerr << "ERROR: Could not retrieve frame " << inputImage << std::endl;
        return 1;
    }
    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn, CV_32F);

    // init kernel
	int MK = 0;
	int NK = 0;
    int kn = MK * NK;
	size_t kernel_bytes = (size_t)(kn)*sizeof(float);
	float kElem_value = 1.0 / kn;
    float *kernel = new float[kn];    // DONE i think size should be kn (5.1) allocate array
    //  initialize kernel to uniform.

	for(int i = 0; i<nc;i++)
	{
		kernel[i] = kElem_value;
	} 

    // get image dimensions
    int w = mIn.cols;         // width
    int h = mIn.rows;         // height
    int nc = mIn.channels();  // number of channels
	int img_size = w*h*nc;
    size_t nbytes = (size_t)(img_size)*sizeof(float);
    std::cout << "Image: " << w << " x " << h << std::endl;

    // initialize CUDA context
    // cudaDeviceSynchronize();

    // ### Set the output image format
    cv::Mat mOut(h,w,mIn.type());  // grayscale or color depending on input image, nc layers

    // ### Allocate arrays
    // allocate raw input image array
    float *imgIn = new float[img_size];
    float *imgOut = new float[img_size];

    // allocate arrays on GPU
    // float *d_imgIn = NULL;
    // float *d_imgOut = NULL;
    // float *d_kernel = NULL;
	// cudaMalloc(&d_imgIn,nbytes);
	// cudaMalloc(&d_imgOut,nbytes);
	// cudaMalloc(&d_kernel,kernel_bytes);

	// convert range of each channel to [0,1]
	mIn /= 255.0f;
	// init raw input image array (and convert to layered)
	convertMatToLayered (imgIn, mIn);

	// TODO IMPLEMENT THESE FUNCTIONS
	// 1. pre-process: pad image
	// 2. perform blind de-convolution

	// padImage(imgIn);
	// computeDeconvolution(imgOut, imgIn, kernel, w, h, nc);


	//cudaMemcpy(imgOut,d_imgOut,nbytes,cudaMemcpyDeviceToHost);

	// show input image
	//showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

	// show output image: first convert to interleaved opencv format from the layered raw array
	//convertLayeredToMat(mOut, imgOut);
	//showImage("Output", mOut, 100+w+40, 100);


    // ### Free allocated arrays
	// cudaFree(d_imgIn);
	// cudaFree(d_imgOut);
	// cudaFree(d_kernel);

    delete[] imgIn;
	delete[] imgOut;
    delete[] kernel;

    // close all opencv windows
    cv::destroyAllWindows();

    return 0;
}



