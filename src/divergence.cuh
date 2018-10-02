#ifndef DIVERGENCE_CUH
#define DIVERGENCE_CUH


void computeDivergenceCuda(float *div, float *dx, float *dy, const float *imgIn, size_t w, size_t h, size_t nc); 


void computeDiffOperatorsCuda(float *d_div, 
                              float *d_dx_fw, float *d_dy_fw,
                              float *d_dx_bw, float *d_dy_bw,
                              float *d_dx_mixed, float *d_dy_mixed, 
                              const float *d_imgIn, size_t w, size_t h, size_t nc);

void computeDivergence(float *div, float *dx, float *dy, const float *imgIn, size_t w, size_t h, size_t nc);

#endif  //  DIVERGENCE_CUH
