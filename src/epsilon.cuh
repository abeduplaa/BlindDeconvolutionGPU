#ifndef EPSILON_CUH
#define EPSILON_CUH


// void computeEpsilonU();


float computeMaxElem(const float *array, const float size);

void computeAbsArray(float *absarray, const float *array, size_t size);

float computeEpsilon(const float *imgIn, const float *gradU, const int size, const float smallnum);

void computeEpsilonGlobalMemCuda(const float *imgIn, const float *gradU, const int size, const float smallnum);

float computeEpsilonCuda(const float *a, const float *grad, const int size, const float smallnum, cublasHandle_t handle);


#endif  //  EPSILON_CUH
