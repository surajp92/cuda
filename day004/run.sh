# module load CUDA
nvcc -arch sm_80 simpleDivergence.cu -o simpleDivergence
./simpleDivergence 