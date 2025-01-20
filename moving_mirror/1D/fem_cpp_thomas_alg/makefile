CXX = g++ -O3
CFLAGS = #-Wall
prog_file = fem_matrices.o moving.o exact.o main.o

.cpp.o:
	$(CXX)	-c	$(CFLAGS)	-c	$<
my_exe:	$(prog_file)
	$(CXX)	$(CFLAGS)	$(prog_file)	-o	my_exe
clean:
	rm	*_exe	*.o	*~
