#ifndef DIVERGENCE_CUH
#define DIVERGENCE_CUH


void computeDivergenceCuda(float *div, float *dx, float *dy, const float *imgIn, size_t w, size_t h, size_t nc); 
void computeDivergence(float *div, float *dx, float *dy, const float *imgIn, size_t w, size_t h, size_t nc);

#endif  //  DIVERGENCE_CUH
