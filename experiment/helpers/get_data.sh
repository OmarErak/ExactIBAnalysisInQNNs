#!/bin/sh

mkdir -p data
cd data

# Get mnist
mkdir -p mnist
wget -O mnist/mnist.data.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.bz2
wget -O mnist/mnist.data.t.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.t.bz2
bzip2 -d mnist/mnist.data.bz2
bzip2 -d mnist/mnist.data.t.bz2

# Get fashion-mnist
mkdir -p fashion-mnist
wget -O fashion-mnist/train-images-idx3-ubyte.gz http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
wget -O fashion-mnist/train-labels-idx1-ubyte.gz http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
wget -O fashion-mnist/t10k-images-idx3-ubyte.gz http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
wget -O fashion-mnist/t10k-labels-idx1-ubyte.gz http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz

# Decompress fashion-mnist files
gunzip fashion-mnist/train-images-idx3-ubyte.gz
gunzip fashion-mnist/train-labels-idx1-ubyte.gz
gunzip fashion-mnist/t10k-images-idx3-ubyte.gz
gunzip fashion-mnist/t10k-labels-idx1-ubyte.gz
