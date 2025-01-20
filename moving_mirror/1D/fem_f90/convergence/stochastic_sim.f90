program stochastic_sim
use mpi
use fem_moving_routines
IMPLICIT NONE
!declaring variables 
double precision :: a,b,T,beta,k
integer :: ierror
integer :: my_rank,num_cores,h_vec(6)
integer :: powpow,s_dim

CALL MPI_Init(ierror)
CALL MPI_Comm_rank(MPI_COMM_WORLD,my_rank,ierror) !defining the variable for the rank of the cores 
CALL MPI_COMM_Size(MPI_COMM_WORLD,num_cores,ierror)

!Reading in input for method
if(my_rank == 0)then
   OPEN(UNIT=10,FILE="input.txt")
   READ(10,*) h_vec   
   READ(10,*) T
   READ(10,*) beta
   READ(10,*) k
   READ(10,*) a
   READ(10,*) b
   READ(10,*) powpow
   READ(10,*) s_dim
   close(10)
end if 

CALL MPI_Bcast(T,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(a,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(b,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(beta,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(k,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(h_vec,6,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(powpow,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(s_dim,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)

print *, 'Computing the EOC of the 1D moving mirror problem'

call runner(T,beta,h_vec,k,s_dim,a,b,powpow)

CALL MPI_Finalize(ierror)

end program stochastic_sim
