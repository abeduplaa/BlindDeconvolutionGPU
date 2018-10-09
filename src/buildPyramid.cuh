#ifndef BUILDPYRAMID_CUH
#define BUILDPYRAMID_CUH


int kernelDim(int in, const float scaleMultiplier, const int smallestScale);

int imageDim(int in, const float factor);

int pyramidScale(const int m, const int n, const int smallestScale, 
    const float scaleMultiplier, const float lambdaMultiplier, const float finalLambda, const float largestLambda);

void buildPyramid1(int *wP, int *hP, int *mP, int *nP, float *lambdas,
    const int w, const int h, const int m, const int n,
    const int smallestScale, const float scaleMultiplier, 
    const float lambdaMultiplier, const float lambda, const int pyramidSize);


// void resizeGlobalMemCuda(const float *output, const float *input);

#endif  //  BUILDPYRAMID_CUH
