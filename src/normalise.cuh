#ifndef NORMALISE_H
#define NORMALISE_H

#include <iostream>
#include "cublas_v2.h"

void normaliseGlobalMemCuda(float* kernel, const int m, const int n);

#endif
