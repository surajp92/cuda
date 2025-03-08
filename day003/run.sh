# nvcc checkThreadIndex.cu -o checkThreadIndex
# ./checkThreadIndex

nvcc -arch sm_80 sumMatrixOnGPU.cu -o sum2D
./sum2D