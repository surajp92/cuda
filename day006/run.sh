# module load CUDA
rm -f reduceIntegerNested
nvcc -arch sm_80 reduceIntegerNested.cu -o reduceIntegerNested -lcudadevrt --relocatable-device-code true
./reduceIntegerNested 