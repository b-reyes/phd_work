program stochastic_sim
use mpi
use fem_routines
implicit none 

!declaring variables 
double precision :: a,b,e,f,ggg,hhh,T,x_start,x_end,o,p,gam_mlmc,alpha
integer :: grids(7),N_pow,M_ex,N_ex,L_first,max_ell
integer :: L,shift,option,N_vals_vec(5)
integer :: ierror,t_pard,s_dim
integer :: my_rank,num_cores,m_it_pow

CALL MPI_Init(ierror)
CALL MPI_Comm_rank(MPI_COMM_WORLD,my_rank,ierror) !defining the variable for the rank of the cores 
CALL MPI_COMM_Size(MPI_COMM_WORLD,num_cores,ierror)

print *, 'hi hi'

!Reading in input for method
if(my_rank == 0)then
   OPEN(UNIT=10,FILE="input.txt")
   READ(10,*) option
   READ(10,*) grids
   OPEN(UNIT=11,FILE="inputt.txt")
   READ(11,*) T
   close(11)
   READ(10,*) a
   READ(10,*) b
   READ(10,*) e
   READ(10,*) f
   READ(10,*) ggg
   READ(10,*) hhh
   READ(10,*) o 
   READ(10,*) p
   READ(10,*) L
   READ(10,*) shift
   READ(10,*) x_start
   READ(10,*) x_end
   READ(10,*) t_pard
   READ(10,*) N_pow
   READ(10,*) N_ex
   READ(10,*) M_ex
   READ(10,*) m_it_pow
   READ(10,*) s_dim
   READ(10,*) N_vals_vec
   READ(10,*) gam_mlmc
   READ(10,*) L_first
   READ(10,*) max_ell
   READ(10,*) alpha
   close(10)
end if 
CALL MPI_Bcast(option,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(grids,7,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(T,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(a,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(b,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(e,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(f,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(o,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(p,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(ggg,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(hhh,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(L,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(shift,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(x_start,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(x_end,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(t_pard,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(N_pow,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(N_ex,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(M_ex,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(m_it_pow,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(s_dim,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(N_vals_vec,5,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(L_first,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(max_ell,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(gam_mlmc,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(alpha,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)

if(option == 0)then 
   print *, 'deterministic version'
   call runner(T,grids,L,x_start,x_end,a,b,e,f,ggg,hhh,o,p,shift,t_pard)
elseif(option == 1)then
   print *, 'grapher version'
   call grapher(T,grids,L,x_start,x_end,a,b,e,f,ggg,hhh,o,p,shift,t_pard)   
end if 


CALL MPI_Finalize(ierror)

end program stochastic_sim
