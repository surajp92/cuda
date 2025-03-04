#!/bin/bash

echo "Removing old hello"
rm -f hello

echo "Compiling and running hello.cu"
nvcc -arch sm_90 hello.cu -o hello

echo "Running hello"
./hello