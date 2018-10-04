#!/bin/bash

#noisy.jpg
#postcard.png

DATA_FOLDER=../sequential/data
#IMAGE=gaudi.png
IMAGE=postcard.png
EXE_FILE=./build/deblur

if [ ! -f $EXE_FILE ]; then
    echo "Good job! you forgot to compile your awesome code"
    echo "Compile it again and come back"
else
    echo $EXE_FILE --image $DATA_FOLDER/$IMAGE
    $EXE_FILE --image=$DATA_FOLDER/$IMAGE $1 $2 $3 $4
fi
