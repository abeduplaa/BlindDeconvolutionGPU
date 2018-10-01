#ifndef EPSILON_CUH
#define EPSILON_CUH


void computeEpsilonU();

// void computeEpsilonK();

float computeMaxElem(const float *array, const float size);

void computeAbsArray(float *array, size_t size);

float computeEpsilonU(const float *imgIn, const float *gradU, const float size)


#endif  //  EPSILON_CUH
