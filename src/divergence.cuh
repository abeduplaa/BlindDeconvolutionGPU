#ifndef DIVERGENCE_CUH
#define DIVERGENCE_CUH


void computeDiffOperatorsCuda(float *d_div, 
                              float *d_dx_fw, float *d_dy_fw,
                              float *d_dx_bw, float *d_dy_bw,
                              float *d_dx_mixed, float *d_dy_mixed, 
                              const float *d_imgIn, const int w, const int h, const int nc,
                              const float lamda, const float eps);

#endif  //  DIVERGENCE_CUH
