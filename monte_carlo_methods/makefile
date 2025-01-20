FC = mpif90 -mkl
CXX = g++
FFLAGS= -warn
FC2=mpiifort -mkl



.SUFFIXES: .f90

oned_mcf = oned_mov_mod.o numerical_mod.o mc_mod.o workspace.o stochastic_sim.o
oned_qmcf = oned_mov_mod.o numerical_mod.o qmc_mod.o workspace.o stochastic_sim.o
oned_qmcc = digitalseq_b2g.o
oned_mlmcf = oned_mov_mod.o numerical_mod.o mlmc_mod.o workspace.o stochastic_sim.o
oned_mlqmcf = oned_mov_mod.o numerical_mod.o mlqmc_mod.o workspace.o stochastic_sim.o
oned_mlmctstf = oned_mov_mod.o numerical_mod.o test_mlmc_mod.o workspace.o stochastic_sim.o
twod_mcf = mkl_pardiso.o twod_ac_mod.o numerical_mod.o mc_mod.o workspace.o stochastic_sim.o
twod_qmcf = mkl_pardiso.o twod_ac_mod.o numerical_mod.o qmc_mod.o workspace.o stochastic_sim.o
twod_mlmcf = mkl_pardiso.o twod_ac_mod.o numerical_mod.o mlmc_mod.o workspace.o stochastic_sim.o
twod_mlmctstf = mkl_pardiso.o twod_ac_mod.o numerical_mod.o test_mlmc_mod.o workspace.o stochastic_sim.o
twod_mlqmcf = mkl_pardiso.o twod_ac_mod.o numerical_mod.o mlqmc_mod.o workspace.o stochastic_sim.o
threed_mcf = mkl_pardiso.o threed_ac_mod.o numerical_mod.o mc_mod.o workspace.o stochastic_sim.o
threed_qmcf = mkl_pardiso.o threed_ac_mod.o numerical_mod.o qmc_mod.o workspace.o stochastic_sim.o
threed_mlmcf = mkl_pardiso.o threed_ac_mod.o numerical_mod.o mlmc_mod.o workspace.o stochastic_sim.o
threed_mlmctstf = mkl_pardiso.o threed_ac_mod.o numerical_mod.o test_mlmc_mod.o workspace.o stochastic_sim.o
threed_mlqmcf = mkl_pardiso.o threed_ac_mod.o numerical_mod.o mlqmc_mod.o workspace.o stochastic_sim.o

.f90.o:
	$(FC)	-c	$(FFLAGS)	-c	$<
.cpp.o:
	$(CXX)	-c	$(CFLAGS)	-c	$<
my_exe: $(FOS)	$(COS)
	$(FC)	$(FOS)	$(COS)	$(FFLAGS) -lstdc++	-o	my_exe
onedmc_exe:	$(oned_mcf)
	$(FC)	$(oned_mcf)	$(FFLAGS)	-o onedmc_exe
twodmc_exe:	$(twod_mcf)
	$(FC)	$(twod_mcf)	$(FFLAGS)	-o twodmc_exe
threedmc_exe:	$(threed_mcf)
	$(FC)	$(threed_mcf)	$(FFLAGS)	-o threedmc_exe
onedqmc_exe:	$(oned_qmcf)	$(oned_qmcc)
	$(FC2)	$(oned_qmcf)	$(oned_qmcc)	$(FFLAGS)	-lstdc++	-o	onedqmc_exe
twodqmc_exe:	$(twod_qmcf)	$(oned_qmcc)
	$(FC2)	$(twod_qmcf)	$(oned_qmcc)	$(FFLAGS)	-lstdc++ 	-o	twodqmc_exe
threedqmc_exe:	$(threed_qmcf)	$(oned_qmcc)
	$(FC2)	$(threed_qmcf)	$(oned_qmcc)	$(FFLAGS)	-lstdc++	-o	threedqmc_exe
onedmlmc_exe:	$(oned_mlmcf)
	$(FC)	$(oned_mlmcf)	$(FFLAGS)	-o	onedmlmc_exe
onedmlmctst_exe:   $(oned_mlmctstf)
	$(FC)	$(oned_mlmctstf)   $(FFLAGS)       -o      onedmlmctst_exe
onedmlqmc_exe:	$(oned_mlqmcf)	$(oned_qmcc)
	$(FC2)	$(oned_mlqmcf)	$(oned_qmcc)	$(FFLAGS)	-lstdc++	-o	onedmlqmc_exe
twodmlmc_exe:	$(twod_mlmcf)
	$(FC)	$(twod_mlmcf)	$(FFLAGS)	-o	twodmlmc_exe
twodmlmctst_exe:   $(twod_mlmctstf)
	$(FC)   $(twod_mlmctstf)   $(FFLAGS)       -o      twodmlmctst_exe
twodmlqmc_exe:	$(twod_mlqmcf)	$(oned_qmcc)
	$(FC2)  $(twod_mlqmcf)	$(oned_qmcc)	$(FFLAGS)	-lstdc++	-o	twodmlqmc_exe
threedmlmc_exe:	$(threed_mlmcf)
	$(FC)	$(threed_mlmcf)	$(FFLAGS)	-o	threedmlmc_exe
threedmlmctst_exe: $(threed_mlmctstf)
	$(FC)   $(threed_mlmctstf) $(FFLAGS)       -o      threedmlmctst_exe
threedmlqmc_exe:	$(threed_mlqmcf)	$(oned_qmcc)
	$(FC2)	$(threed_mlqmcf)	$(oned_qmcc)	$(FFLAGS)	-lstdc++	-o	threedmlqmc_exe
clean:
	rm	-rf	*_exe	*.o	 *mod 	*~




