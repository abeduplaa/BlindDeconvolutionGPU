#include <iostream>
#include <vector>
#include <math.h>

#include "buildPyramid.cuh"


/////////////////

// GPU FUNCTIONS:
__global__ 
void resizeCudaKernel( unsigned char* input,
    unsigned char* output,
    const int outputWidth,
    const int outputHeight,
    const int inputWidthStep,
    const int outputWidthStep,
    const float pixelGroupSizeX,
    const float pixelGroupSizeY,
    const int inputChannels)
{
//2D Index of current thread
const int outputXIndex = blockIdx.x * blockDim.x + threadIdx.x;
const int outputYIndex = blockIdx.y * blockDim.y + threadIdx.y;

//Only valid threads perform memory I/O
if((outputXIndex<outputWidth) && (outputYIndex<outputHeight))
{
    // Starting location of current pixel in output
    int output_tid  = outputYIndex * outputWidthStep + (outputXIndex * inputChannels);

    // Compute the size of the area of pixels to be resized to a single pixel
    const float pixelGroupArea = pixelGroupSizeX * pixelGroupSizeY;

    // Compute the pixel group area in the input image
    const int intputXIndexStart = int(outputXIndex * pixelGroupSizeX);
    const int intputXIndexEnd = int(intputXIndexStart + pixelGroupSizeX);
    const float intputYIndexStart = int(outputYIndex * pixelGroupSizeY);
    const float intputYIndexEnd = int(intputYIndexStart + pixelGroupSizeY);

    if(inputChannels==1) { // grayscale image
        float channelSum = 0;
        for(int intputYIndex=intputYIndexStart; intputYIndex<intputYIndexEnd; ++intputYIndex) {
            for(int intputXIndex=intputXIndexStart; intputXIndex<intputXIndexEnd; ++intputXIndex) {
                int input_tid = intputYIndex * inputWidthStep + intputXIndex;
                channelSum += input[input_tid];
            }
        }
        output[output_tid] = static_cast<unsigned char>(channelSum / pixelGroupArea);
    } else if(inputChannels==3) { // RGB image
        float channel1stSum = 0;
        float channel2stSum = 0;
        float channel3stSum = 0;
        for(int intputYIndex=intputYIndexStart; intputYIndex<intputYIndexEnd; ++intputYIndex) {
            for(int intputXIndex=intputXIndexStart; intputXIndex<intputXIndexEnd; ++intputXIndex) {
                // Starting location of current pixel in input
                int input_tid = intputYIndex * inputWidthStep + intputXIndex * inputChannels;
                channel1stSum += input[input_tid];
                channel2stSum += input[input_tid+1];
                channel3stSum += input[input_tid+2];
            }
        }
        output[output_tid] = static_cast<unsigned char>(channel1stSum / pixelGroupArea);
        output[output_tid+1] = static_cast<unsigned char>(channel2stSum / pixelGroupArea);
        output[output_tid+2] = static_cast<unsigned char>(channel3stSum / pixelGroupArea);
    } else { 
    }
}
}


void resizeGlobalMemCuda(const float *output, const float *input)
{
    // allocate block and grid size
	dim3 block(32, 8, 1);
    dim3 grid = computeGrid2D(block, w - m + 1, h - n + 1);
    
	//Specify a reasonable block size
	// const dim3 block(16,16);
 
	//Calculate grid size to cover the whole image
	// const dim3 grid((output.cols + block.x - 1)/block.x, (output.rows + block.y - 1)/block.y);
 
	// Calculate how many pixels in the input image will be merged into one pixel in the output image
	const float pixelGroupSizeY = float(input.rows) / float(output.rows);
	const float pixelGroupSizeX = float(input.cols) / float(output.cols);
 
	//Launch the size conversion kernel
	resizeCudaKernel<<<grid,block>>>(d_input,d_output,output.cols,output.rows,input.step,output.step, pixelGroupSizeX, pixelGroupSizeY, input.channels());
 
	timer.Stop();
	printf("Own Cuda code ran in: %f msecs.\n", timer.Elapsed());
 
	//Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");
 
	//Copy back data from destination device meory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(),d_output,outputBytes,cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");
 
	//Free the device memory
	SAFE_CALL(cudaFree(d_input),"CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output),"CUDA Free Failed");
	//SAFE_CALL(cudaDeviceReset(),"CUDA Device Reset Failed");
}


///////
//CPU FUNCTIONS


int kernelDim(int in, const float scaleMultiplier, const int smallestScale)
{
    // calculate out
    int out = round( (float)in / scaleMultiplier );
    
    // check if dimension is even
    out = (out%2 == 0) ? (out-1) : out;

    // check if dimension same as last dimension
    out = (out == in) ? (out-2) : out;

    // check if dimension is  smaller than smallest value
    out = (out < smallestScale) ? smallestScale : out;
    
    return out;
}

int imageDim(int in, const float factor)
{
    int out = round( (float)in / factor );

    // check if dimension is even
    out = (out%2 == 0) ? (out-1) : out;

    return out;

}

int pyramidScale(const int m, const int n, const float smallestScale, 
    const float scaleMultiplier, const float lambdaMultiplier)
{
    //dummy vars:
    int m1 = m;
    int n1 = n;
    int l1 = 0;
    int pyramidSize = 0;

    while( (m1 > smallestScale) && (n1 > smallestScale) 
    && (l1 * lambdaMultiplier < largestLambda) )
    {
        m1 = kernelDim(m1, scaleMultiplier, smallestScale);

        n1 = kernelDim(n1, scaleMultiplier, smallestScale);

        l1 = l1 * lambdaMultiplier;

        pyramidSize += 1;
    }

    return pyramidSize;
}

void buildPyramid(int *wP, int *hP, int *mP, int *nP, float *lambdas, 
    const int w, const int h, const int m, const int n, 
    const int smallestScale, const float lambda, const float finalLambda, 
    const float scaleMultiplier, const float lambdaMultiplier, const int pyramidSize)
{
    float factorW = 0;
    float factorH = 0;

    wP[0] = w;
    hP[0] = h;
    mP[0] = m;
    nP[0] = n;
    lambdas[0] = lambda;
    
    // while( (mP.back() > smallestScale) && (np.back() > smallestScale) 
    // &&  (lambdas.back() * lambdaMultiplier < largestLambda) )
    
    for(int i = 1 ; i < pyramidSize ; i++)
    {
        lambdas[i] = lambdas[i-1] *lambdaMultiplier;

        mP[i] = kernelDim(mP[i - 1], scaleMultiplier, smallestScale);
        nP[i] = kernelDim(nP[i - 1], scaleMultiplier, smallestScale);
        
        factorW = mP[i - 1] / mP[i];
        factorH = nP[i - 1] / nP[i];

        wP[i] = imageDim(wP[i-1] , factorW);
        hP[i] = imageDim(hP[i-1] , factorH);
    }
}



