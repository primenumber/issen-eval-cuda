cgls: solve.o get_input.o subboard.o bit_manipulations.o
	nvcc -o cgls -g -arch=sm_61 -std=c++11 -O2 subboard.o bit_manipulations.o get_input.o solve.o

clean:
	rm *.o
	rm cgls

%.o : %.cu
	nvcc -c -o $@ -g -arch=sm_61 -std=c++11 -O2 $<

%.cubin : %.cu
	nvcc -cubin -arch=sm_61 -std=c++11 -O2 $<

%.o : %.cpp
	g++-6 -c -o $@ -std=c++14 -march=native -mtune=native -O2 -Wall -Wextra $<

solve.o: get_input.hpp board_gpu.cuh
get_input.o: get_input.hpp board.hpp subboard.hpp
subboard.o: subboard.hpp bit_manipulations.hpp
bit_manipulations.o: bit_manipulations.hpp board.hpp
