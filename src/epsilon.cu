#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

#include "epsilon.cuh"


void computeMaxArray(float &maxElem, const float *array, const float *size)
{
    maxElem = std::max_element(array,array+size);
}

void computeAbsArray(float *absArray, const float *array, size_t size)
{
    for(int i = 0; i<size; i++)
    {
        absArray[i] = fabs(array[i]);
    }
}

void computeEpsilonU(float *eps, const float *imgIn, const float *gradU)
{
    5e-3*max(u(:))/max(1e-31,max(max(abs(gradu(:)))));

    float *maxElemU;
    float *maxElemG;
    float *absElemG;
    float lower = 0;

    computeAbsArray(absElemG, gradU, sizeG);
    
    computeMaxArray(maxElemG, absElemG, sizeG);

    if(1e-31 > maxElemG)
    {
        lower = 1e31;
    }else
    {
        lower = 1/maxElemG;
    }

    computeMaxArray(maxElemU, imgIn, sizeU);

    eps = (0.005 * maxElemU ) * lower;

}

void computeEpsilonK()
{
    1e-3*max(k(:))/max(1e-31,max(max(abs(gradk))));
}