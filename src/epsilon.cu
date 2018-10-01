#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include "epsilon.cuh"


float computeMaxElem(const float *array, const float size)
{
    float maxElem = 0;
    
    maxElem = std::max_element(array,array+size);

    return maxElem;
}

//void computeAbsArray(float *absArray, const float *array, size_t size)
void computeAbsArray(float *array, size_t size)
{
    for(int i = 0; i<size; i++)
    {
        //absArray[i] = fabs(array[i]);
        array[i] = fabs(array[i]);        
    }
}

float computeEpsilonU(const float *imgIn, float *gradU, const float size)
{
    // 5e-3*max(u(:))/max(1e-31,max(max(abs(gradu(:)))));

    // float *absElemG;
    float maxGrad = 0;
    float maxElemU = 0;
    float lower = 0;
    float eps = 0;

    computeAbsArray(gradU, size);
    
    maxGrad = computeMaxElem(gradU, size);

    if(1e-31 > maxGrad)
    {
        lower = 1e31;
    }else
    {
        lower = 1/maxGrad;
    }

    maxElemU = computeMaxElem(imgIn, size);

    eps = (0.005 * maxElemU ) * lower;

    return eps;

}

void computeEpsilonK()
{
    //1e-3*max(k(:))/max(1e-31,max(max(abs(gradk))));
}