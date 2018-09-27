#!/bin/bash

#noisy.jpg
#postcard.png

DATA_FOLDER=../sequential/data/
IMAGE=postcard.png
EXE_FILE=./build/deblur

if [ ! -f $EXE_FILE ]; then
    echo "Good job! you forgot to compile your awesome code"
    echo "Compile it again and come back"
else
    $EXE_FILE -i $DATA_FOLDER/$IMAGE
fi
