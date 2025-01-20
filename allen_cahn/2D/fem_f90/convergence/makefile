FM90 = mpif90 -mkl
move = mkl_pardiso.f90 fem_routines.f90 stochastic_sim.f90

all:my_exe

my_exe:$(move)
	$(FM90)	$(move)	-o	$@
clean:
	rm	*_exe	*.mod	*~