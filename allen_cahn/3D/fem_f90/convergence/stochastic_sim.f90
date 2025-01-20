program stochastic_sim
use mpi
use fem_routines
use mkl_pardiso
implicit none

!declaring variables                                                  
double precision :: a,b,c,d,e,f,ggg,hhh,T,ex_qoi,gam_mlmc,x_start,x_end,al_val
integer,allocatable,dimension(:) :: grids_det,n_array,grids_mlmc
integer :: L,shift,L_first
integer :: option,ierror,m_it_pow,s_dim
integer :: my_rank,num_cores,N_eps_mlmc,N_samp_mc
integer :: N_samp_mlmc,size_n_array,size_grids_det,N_ex_mc,size_grids_mlmc
integer :: len_text_file,pthr
character(len=:),allocatable :: str

CALL MPI_Init(ierror)
CALL MPI_Comm_rank(MPI_COMM_WORLD,my_rank,ierror)
CALL MPI_COMM_Size(MPI_COMM_WORLD,num_cores,ierror)

!Reading in input for method                                           
if(my_rank == 0)then
   OPEN(UNIT=10,FILE="input.txt")
   READ(10,*) option
   READ(10,*) size_n_array
   ALLOCATE(n_array(size_n_array))
   READ(10,*) n_array
   READ(10,*) size_grids_det
   ALLOCATE(grids_det(size_grids_det))
   READ(10,*) grids_det
   READ(10,*) N_ex_mc
   READ(10,*) size_grids_mlmc
   ALLOCATE(grids_mlmc(size_grids_mlmc))
   READ(10,*) grids_mlmc
   READ(10,*) N_eps_mlmc
   READ(10,*) N_samp_mc
   READ(10,*) N_samp_mlmc
   READ(10,*) ex_qoi
   READ(10,*) T
   READ(10,*) a
   READ(10,*) b
   READ(10,*) c
   READ(10,*) d
   READ(10,*) e
   READ(10,*) f
   READ(10,*) ggg
   READ(10,*) hhh
   READ(10,*) L
   READ(10,*) shift
   READ(10,*) x_start
   READ(10,*) x_end
   READ(10,*) gam_mlmc
   READ(10,*) L_first
   READ(10,*) al_val
   READ(10,*) pthr
   READ(10,*) m_it_pow
   READ(10,*) s_dim
   close(10)
end if

CALL MPI_Bcast(size_n_array,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(size_grids_det,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(size_grids_mlmc,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)

if(my_rank /= 0)then
   ALLOCATE(n_array(size_n_array))
   ALLOCATE(grids_det(size_grids_det))
   ALLOCATE(grids_mlmc(size_grids_mlmc))
end if

CALL MPI_Bcast(option,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(n_array,size_n_array,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(grids_det,size_grids_det,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(N_ex_mc,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(grids_mlmc,size_grids_mlmc,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(N_eps_mlmc,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(N_samp_mc,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(N_samp_mlmc,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(ex_qoi,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(T,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(a,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(b,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(c,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(d,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(e,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(f,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(ggg,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(hhh,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(L,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(shift,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(x_start,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(x_end,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(gam_mlmc,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(L_first,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(al_val,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(pthr,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(m_it_pow,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(s_dim,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
CALL MPI_BARRIER(MPI_COMM_WORLD,ierror)

!choosing which method to choose                                       
if(option == 0)then
   DEALLOCATE(n_array,grids_mlmc)
   print *, 'deterministic version'
   call runner(T,grids_det,L,x_start,x_end,a,b,c,d,e,f,ggg,hhh,shift,pthr)
   DEALLOCATE(grids_det)
end if

CALL MPI_Finalize(ierror)

end program stochastic_sim
