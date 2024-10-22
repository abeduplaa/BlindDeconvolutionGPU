# BlindDeconvolutionGPU

**Speeding up blind deconvolution of a blurred image by using NVIDIA GPUs.**

Provided with the matlab code from the paper [**Total Variation Blind Deconvolution: The Devil is in the Details**](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Perrone_Total_Variation_Blind_2014_CVPR_paper.pdf),
BlindDeconvolutionGPU is a sped up implementation of the proposed algorithm with NVIDIA GPUs. 


The speed up algorithm has been implemented in C with CUDA libraries. This implementation is part of the practical course: GPU Programming in Computer Vision offered at the Technical University of Munich (TUM).

The proposed algorithm was developed by the Computer Vision Group at the University of Bern. The code has been included in this repository under the '/sequential' folder. The authors of the paper and developers of the sequential code are [Daniele Perrone](perrone@iam.unibe.ch) and [Paolo Favaro](paolo.favaro@iam.unibe.ch).

Requirements:
1. CUDA
2. Open CV
3. CUDNN
4. CUB

## Links

For more information about blind deconvolution, the proposed algorithm, and the authors, please refer to the following:

[Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Perrone_Total_Variation_Blind_2014_CVPR_paper.pdf)

[Computer Vision Group at University of Bern](http://www.cvg.unibe.ch/home/)

[Original Project Page](http://www.cvg.unibe.ch/media/project/perrone/tvdb/index.html)

[Daniele Perrone homepage](https://danieleperrone.com/)

[Paolo Favaro homepage](http://www.cvg.unibe.ch/people/person/3)



