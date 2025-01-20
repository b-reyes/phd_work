program stochastic_sim
use,intrinsic :: iso_c_binding
use mpi
use workspace 

IMPLICIT NONE
!declaring variables  
integer :: option,ierror,N_pow,N0
integer :: s_dim,my_rank,shift_pow
double precision :: eps,gam_mlmc
integer :: n_pows(8),grid_vals(8)
integer :: N_end,M_end,l_first

CALL MPI_Init(ierror)
CALL MPI_Comm_rank(MPI_COMM_WORLD,my_rank,ierror) !defining the variable for the rank of the cores 

!Reading in input for method
if(my_rank == 0)then
   OPEN(UNIT=10,FILE="input.txt")
   READ(10,*) option

   if(option == 0)then
      READ(10,*) N_pow
      READ(10,*) s_dim
      READ(10,*) N_end
      READ(10,*) M_end
      close(10)
   else if(option == 1)then
      READ(10,*) N_pow
      READ(10,*) s_dim
      READ(10,*) N_end
      READ(10,*) M_end
      READ(10,*) shift_pow
      close(10)
   else if(option == 2)then
      READ(10,*) eps
      READ(10,*) s_dim
      READ(10,*) n_pows
      READ(10,*) grid_vals
      READ(10,*) gam_mlmc
      READ(10,*) l_first
   else if(option == 3)then
      READ(10,*) eps
      READ(10,*) s_dim
      READ(10,*) n_pows
      READ(10,*) grid_vals
      READ(10,*) l_first
      READ(10,*) shift_pow
   else if(option == 4)then
      READ(10,*) N0
      READ(10,*) s_dim
      READ(10,*) grid_vals
      READ(10,*) l_first
   end if
end if 

CALL MPI_Bcast(option,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)

if((option == 1) .OR. (option == 0))then
   CALL MPI_Bcast(N_pow,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
   CALL MPI_Bcast(N_end,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
   CALL MPI_Bcast(M_end,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
else if((option == 3) .OR. (option == 2))then
   CALL MPI_Bcast(eps,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
   CALL MPI_Bcast(n_pows,8,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
   CALL MPI_Bcast(grid_vals,8,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
   CALL MPI_Bcast(l_first,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
else if(option == 4)then
   CALL MPI_Bcast(N0,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
   CALL MPI_Bcast(grid_vals,8,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
   CALL MPI_Bcast(l_first,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
end if

CALL MPI_Bcast(s_dim,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)

if((option == 1) .OR. (option == 3))then
   CALL MPI_Bcast(shift_pow,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
else if(option == 2)then
   CALL MPI_Bcast(gam_mlmc,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
end if

if(my_rank == 0)then                                             
