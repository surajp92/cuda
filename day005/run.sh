# module load CUDA
nvcc -arch sm_80 reduceInteger.cu -o reduceInteger
./reduceInteger 