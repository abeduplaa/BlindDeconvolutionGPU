#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

#include "epsilon.cuh"
#include "helper.cuh"



__global__
void computeEpsilonGlobalMemKernel(float *eps, const float *a, const int a_i, const float *grad, const int grad_i, const float smallnum)
{
	// int idx = threadIdx.x + blockIdx.x*blockDim.x;
	// int idy = threadIdx.y + blockIdx.y*blockDim.y;
    // calling cuda kernel
    *eps = (smallnum * a[a_i]) * ( ( grad[grad_i] < 1e31) ? (1.0/grad[grad_i]) : (1e-31) );
    
}


void computeEpsilonGlobalMemCuda(float *eps, cublasHandle_t handle, const float *a, const float *grad, const int size, const float smallnum)
{
    	// allocate block and grid size
	dim3 block(32, 8, 1);
	//dim3 grid = computeGrid2D(block, m - h + 1, n - w + 1);
	dim3 grid = computeGrid2D(block, 8, 8);

    //initialize indices:
    int a_i = NULL;
    int grad_i = NULL;
    
    // call cublas functions:
    absMaxIdCUBLAS(handle, size, a, 1, &a_i);

    absMaxIdCUBLAS(handle, size, grad, 1, &grad_i);
	
	//calling cuda kernel
    computeEpsilonGlobalMemKernel <<<grid,block>>> (eps, a, a_i, grad, grad_i, smallnum);
}



// CPU FUNCTIONS

float computeMaxElem(const float *array,const float size)
{
    float maxElem = array[0];
	
	for(int i=1; i<size; i++)
	{
		if (array[i]>array[i-1])
		{
			maxElem = array[i];
		}
	}

	//maxElem = *std::max_element(array,array+size);

    return maxElem;
}

//void computeAbsArray(float *absArray, const float *array, size_t size)
void computeAbsArray(float *absarray, const float *array, size_t size)
{
    for(int i = 0; i<size; i++)
    {
        //absArray[i] = fabs(array[i]);
        absarray[i] = fabs(array[i]);        
    }
}

float computeEpsilon(const float *imgIn, const float *gradU, const int size, const float smallnum)
{
    // 5e-3*max(u(:))/max(1e-31,max(max(abs(gradu(:)))));

    // float *absElemG;
    float maxGrad = 0;
    float maxElemU = 0;
    float lower = 0;
    float eps = 0;
	float *absgradU = new float[size];

    computeAbsArray(absgradU, gradU, size);
    
    maxGrad = computeMaxElem(absgradU, size);

    if(1e-31 > maxGrad)
    {
        lower = 1e31;
    }else
    {
        lower = 1/maxGrad;
    }

    maxElemU = computeMaxElem(imgIn, size);

    eps = (smallnum * maxElemU ) * lower;

    return eps;

}

float computeEpsilonCuda(cublasHandle_t handle, const float *a, const float *grad, const int size, const float smallnum)
{
    //initialize indices:
    int a_i = NULL;
    int grad_i = NULL;
    
    // call cublas functions:
    absMaxIdCUBLAS(handle, size, a, 1, &a_i);

    absMaxIdCUBLAS(handle, size, grad, 1, &grad_i);


    // calling cuda kernel
    //return (smallnum * a[a_i]) * ( ( grad[grad_i] < 1e31) ? (1.0/grad[grad_i]) : (1e-31) );
	return (float)grad_i;
}
