cgls: main.cu
	nvcc -o cgls -arch=sm_61 -std=c++11 -O2 -expt-relaxed-constexpr main.cu
