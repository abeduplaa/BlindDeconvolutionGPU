#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

#include "epsilon.cuh"
#include "helper.cuh"



__global__
void computeEpsilonGlobalMemKernel(float *eps, const float *u, const int u_max_index,
                                   const float *grad, const int grad_max_index, 
                                   const float smallnum) {
    
    // NOTE: assume that u is always greater than zero
    float abs_grad_max = (grad[grad_max_index] > 0.0) ? grad[grad_max_index] 
                                                      : -1.0f * grad[grad_max_index];

    float denominator = max(1e-31, abs_grad_max);
    *eps = -1.0f * (smallnum * u[u_max_index]) / denominator;
}


void computeEpsilonGlobalMemCuda(float *eps, cublasHandle_t handle,
                                 const float *u, const float *grad,
                                 const int size, const float smallnum) {
    // allocate block and grid size
	// TODO: What should these values be?
	/*dim3 block(32, 8, 1);*/
	/*dim3 grid = computeGrid2D(block, 8, 8);*/

    // NOTE: assume that u is greater than zero

    //initialize indices:
    int u_max_index = 0;
    int grad_max_index = 0;
    
    // call cublas functions to get highest value elements:
    cublasIsamax(handle, size, u, 1, &u_max_index); 
    CUDA_CHECK;
    
    cublasIsamax(handle, size, grad, 1, &grad_max_index);
    CUDA_CHECK;

	// subtract one due to BLAS starting at 1
	u_max_index -= 1;
    grad_max_index -= 1;
    
	//calling cuda kernel
    computeEpsilonGlobalMemKernel<<<1,1>>>(eps, u, u_max_index,
                                           grad, grad_max_index,
                                           smallnum);
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
        lower = 1.f/maxGrad;
    }

    maxElemU = computeMaxElem(imgIn, size);
	
	std::cout<<"CPU.. maxU: " << maxElemU << ", maxGrad: " << maxGrad << ", smallnum: "<<smallnum<< "\n";
	
    eps = (smallnum * maxElemU ) * lower;

    return eps;

}

/*
float computeEpsilonCuda(cublasHandle_t handle, const float *a, const float *grad, const int size, const float smallnum)
{
    //initialize indices:
    int a_i = 0;
    int grad_i = 0;
    
    // call cublas functions:
    absMaxIdCUBLAS(handle, size, a, 1, &a_i);

    absMaxIdCUBLAS(handle, size, grad, 1, &grad_i);


    // calling cuda kernel
    //return (smallnum * a[a_i]) * ( ( grad[grad_i] < 1e31) ? (1.0/grad[grad_i]) : (1e-31) );
	return (float)grad_i;
}
*/
