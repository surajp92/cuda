# gcc sumArraysOnHost.c -o sum
# ./sum

# nvcc -arch sm_80 checkDimension.cu -o checkdim
# ./checkdim

nvcc -arch sm_80 sumArraysOnGPU.cu -o sumcuda
./sumcuda
