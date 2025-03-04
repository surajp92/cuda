#!/bin/bash


echo "Compiling and running hello.cu"
rm -f hello

nvcc -arch sm_90 hello.cu -o hello

./hello