#!/bin/sh

trainingfile="train.p"
alexnetfile="bvlc-alexnet.npy"


if [ -f "$trainingfile" ]
then
        echo "Downloaded dataset"
else
        echo "Dataset not not found"
        echo "Downloading data set"
        wget https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580a829f_train/train.p
        echo "Download complete"     
fi

if [ -f "$alexnetfile" ]
then
        echo "Downloaded alext net py"
else
        echo "Dataset not not found"
        echo "Downloading data set"
        wget https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580d880c_bvlc-alexnet/bvlc-alexnet.npy
        echo "Download complete"     
fi




