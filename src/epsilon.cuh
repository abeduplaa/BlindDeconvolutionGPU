#ifndef EPSILON_CUH
#define EPSILON_CUH
#include "cublas_v2.h"

// void computeEpsilonU();


float computeMaxElem(const float *array, const float size);

void computeAbsArray(float *absarray, const float *array, size_t size);

float computeEpsilon(const float *imgIn, const float *gradU, const int size, const float smallnum);

//float computeEpsilonCuda(float *eps, cublasHandle_t handle, const float *a, const float *grad, const int size, const float smallnum);

void computeEpsilonGlobalMemCuda(float *eps, cublasHandle_t handle, const float *a, const float *grad, const int size, const float smallnum);


#endif  //  EPSILON_CUH
