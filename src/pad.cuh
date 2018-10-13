#ifndef PAD_H
#define PAD_H

#include <iostream>

enum BoundaryCondition {replicate, zero, periodic, symmetric};

void padImgCPU(float* imgOut, const float* imgIn, 
        size_t w, size_t h, size_t nc, size_t m, size_t n);

void padImgGlobalMemCuda(float* imgOut, const float* imgIn, 
        int w, int h, int nc, int m, int n, BoundaryCondition& bc); 

void selectBoundaryCondition(char bc, BoundaryCondition& boundary);

#endif
