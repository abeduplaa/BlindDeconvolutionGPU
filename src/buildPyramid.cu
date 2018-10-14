#include <iostream>
#include <vector>
#include <math.h>
#include <opencv2/core/core.hpp>
#include "buildPyramid.cuh"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"


int kernelDim(int in, const float scaleMultiplier, const int smallestScale)
{
    // calculate out
    int out = round( (float)in / scaleMultiplier );
    
    // check if dimension is even
    out = (out%2 == 0) ? (out-1) : out;

    // check if dimension same as last dimension
    out = (out == in) ? (out-2) : out;

    // check if dimension is  smaller than smallest value
    out = (out < smallestScale) ? smallestScale : out;
    
    return out;
}

int imageDim(int in, const float factor)
{	
	//float in1 = (float) in;
    int out = round( in / factor );

    // check if dimension is even
    out = (out%2 == 0) ? (out-1) : out;
    return out;

}

int pyramidScale(const int m, const int n, const int smallestScale,
                 const float scaleMultiplier, const float lambdaMultiplier,
                 const float finalLambda, const float largestLambda) {
    //dummy vars:
    int m1 = m;
    int n1 = n;
    float l1 = finalLambda;
    int pyramidSize = 1;
    
    while( (m1 > smallestScale) && (n1 > smallestScale) 
    && (l1 * lambdaMultiplier < largestLambda) ) {

        m1 = kernelDim(m1, scaleMultiplier, smallestScale);

        n1 = kernelDim(n1, scaleMultiplier, smallestScale);

        l1 = l1 * lambdaMultiplier;

        pyramidSize += 1;
    }

    return pyramidSize;
}

void buildPyramid1(int *wP, int *hP, int *mP, int *nP, float *lambdas, 
    const int w, const int h, const int m, const int n, 
    const int smallestScale, const float scaleMultiplier, 
	const float lambdaMultiplier, const float lambda, const int pyramidSize)
{
    float factorW = 0.f;
    float factorH = 0.f;

    wP[0] = w;
    hP[0] = h;
    mP[0] = m;
    nP[0] = n;
    lambdas[0] = lambda;
    
    
    for(int i = 1 ; i < pyramidSize ; i++)
    {
        lambdas[i] = lambdas[i-1] *lambdaMultiplier;

        mP[i] = kernelDim(mP[i - 1], scaleMultiplier, smallestScale);
        nP[i] = kernelDim(nP[i - 1], scaleMultiplier, smallestScale);
        
        factorW = mP[i - 1]*1.0 / mP[i];
        factorH = nP[i - 1]*1.0 / nP[i];

        wP[i] = imageDim(wP[i-1] , factorW);
        hP[i] = imageDim(hP[i-1] , factorH);
    }
}
