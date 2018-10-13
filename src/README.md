1. Download cuDNN as tarball archive from the official Nvidia website
    cuDNN v7.0.5 Library for Linux (link: https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.0.5/prod/8.0_20171129/cudnn-8.0-linux-x64-v7)

2. extract the archive
	tar xvf <Download>/cudnn-8.0-linux-x64-v7.tgz

3. rename the folder "cuda" to "cuDNN" avoid some conflicts. Copy "cuDNN" folder 
to ~/usr
	mv cuda cuDNN
	mv ./cuDNN ~/usr/

4. open your .bashrc and add the following
    export CU_DNN_PATH=$HOME/usr/cuDNN
    export LD_LIBRARY_PATH=$CU_DNN_PATH/lib64:$LD_LIBRARY_PATH:               
    export C_INCLUDE_PATH=$C_INCLUDE_PATH:$CU_DNN_PATH/include

5. update your changes
	source ~/.bashrc

6. Go to the deblur source folder and create a new directory "build". Compile deblur
	cd <deblur>/src
	mkdir build
	cd build
	cmake ..
	make

7. run the application
    ./deblur --image=<path-to-image> --mk=<uint> --nk=<uint> --iter=<uint> --bc=<p,s,r,z>
