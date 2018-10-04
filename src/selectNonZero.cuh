#ifndef SELECTNONZERO_H
#define SELECTNONZERO_H

#include <iostream>
#include "helper.cuh"

void selectNonZeroGlobalMemCuda(float* kernel, const int m, const int n);

#endif
