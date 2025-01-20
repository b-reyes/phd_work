MODULE mc_mod
USE mpi
use numerical_mod

contains 

subroutine mc(N_pow,s_dim,N_end,M_end)
implicit none
!declaring variables                                                                          
integer :: ierror,my_rank,num_cores,chunks,master = 0,i
double precision :: time1, time2
integer, intent(in) :: N_pow,s_dim,N_end,M_end
integer, allocatable,dimension(:) :: array_chunks
double precision, allocatable,dimension(:) :: temp_err,mc_error
integer :: N_mc
double precision :: P_temp,final_P,P,variance
integer :: rrr,qq
double precision, allocatable, dimension(:) :: rs,P_temp_vec

CALL MPI_Comm_rank(MPI_COMM_WORLD,my_rank,ierror) !rank of the core              
CALL MPI_COMM_Size(MPI_COMM_WORLD,num_cores,ierror) !number of cores = num_cores      
CALL MPI_BARRIER(MPI_COMM_WORLD,ierror)

if(my_rank == master)then
   !starting the timer               
   print *, 'Number of samples:',2**N_pow
   print *, 'Stochastic dimension:',s_dim
   print *, 'N_end: ',N_end
   print *, 'M_end: ',M_end
   print *, 'Number of cores:',num_cores
   time1 = MPI_Wtime()
end if

ALLOCATE(array_chunks(num_cores))    !allocating array to hold chunks in         

!initializing the output quantity of interest                                    
P = 0
N_mc = int(2)**N_pow

if(my_rank == 0)then

   if(N_mc < num_cores)then
      print *, 'USE LESS CORES!!'
      stop
   end if

   chunks = floor(dble(N_mc)/num_cores)
   rrr = mod(N_mc,num_cores) !getting the remainder so we can distribute it to the cores

   !filling the array_chunks with chunks                                         
   array_chunks(1) = chunks

   do i=2,size(array_chunks)
      array_chunks(i) = chunks
   end do

   qq = 1
   do while(rrr /= 0)
      array_chunks(qq) = array_chunks(qq) + 1
      qq = qq + 1
      rrr = rrr - 1
   end do
end if
CALL MPI_Bcast(array_chunks,num_cores,MPI_INTEGER,master,MPI_COMM_WORLD,ierror)

ALLOCATE(P_temp_vec(array_chunks(my_rank + 1)))

ALLOCATE(rs(s_dim))

!initialize all constant variables for numerical model 
call numerical_model_init

do i = 1,array_chunks(my_rank+1)

   call rand_vec(rs,s_dim,my_rank,dble(0),dble(1.0))

   call numerical_model(rs,P_temp,N_end,M_end)

   P_temp_vec(i) = P_temp
   P = P + P_temp

end do

DEALLOCATE(rs)

call numerical_model_fin

CALL MPI_BARRIER(MPI_COMM_WORLD,ierror)


call MPI_REDUCE(P,final_P,1,MPI_DOUBLE_PRECISION,MPI_SUM,master,MPI_COMM_WORLD,ierror)

if(my_rank == master)then
   final_P = final_P/dble(N_mc)
end if 

CALL MPI_Bcast(final_P,1,MPI_DOUBLE_PRECISION,master,MPI_COMM_WORLD,ierror)

ALLOCATE(temp_err(array_chunks(my_rank + 1)))

if(my_rank == 0)then
   Allocate(mc_error(array_chunks(my_rank + 1)))
end if

temp_err = sum((P_temp_vec - final_P)**2)

call MPI_REDUCE(temp_err,mc_error,array_chunks(my_rank+1),MPI_DOUBLE_PRECISION,MPI_SUM,master,MPI_COMM_WORLD,ierror)

DEALLOCATE(temp_err)

if(my_rank == 0)then
   time2 = MPI_Wtime()
   variance = sum(mc_error)/dble(N_mc*(N_mc - 1))
   print *, 'E[Q]: ', final_P
   print *, 'S_E[Q]: ', sqrt(variance)
   print *, 'elapsed time in seconds (mult. by num_cores): ', (time2 - time1)*dble(num_cores)
   DEALLOCATE(mc_error)
end if

DEALLOCATE(P_temp_vec,array_chunks)

end subroutine mc

!creating a random vector, r_k where the values are between 1 and 5, and j is the length of r_k
subroutine rand_vec(r_k,j,k,a,b)
implicit none
!declaring variables                                             
double precision, intent(in) :: a,b
double precision,allocatable, dimension(:),intent(inout) :: r_k
integer :: size,i,time
integer, intent(in) :: j,k
integer,allocatable,dimension(:) :: seed

size = j+12          !setting the size of the vector of seeds, 12 is the smallest seed you can have 
allocate(seed(size)) !seed vector                                                   

!Setting the seeds with respect to the system time and the rank              
do i = 1,size
call system_clock(time)
 seed(i) = time*(k+1)    !has to be plus one because of rank=0                         
end do

CALL random_seed(size=size)   !applying the size of the seeds                       
CALL random_seed(put=seed)    !applying the seed to random_seed                         

!creating a vector of random numbers                                                   
do i = 1,j
CALL random_number(r_k)
end do

!putting the numbers in the range [a,b]                                           
do i = 1,j
   r_k(i) = r_k(i)*(b - a) + a
end do

deallocate(seed)   !deallocating the seed, we don't need it anymore                       

end subroutine rand_vec

function endfun(chunks,rank)
implicit none

integer :: endfun,rank,i
integer, allocatable,dimension(:) :: chunks

endfun = 0

do i = 1,rank+1
   endfun = endfun + chunks(i)
end do

end function

function startfun(chunks,rank)
implicit none

integer :: startfun,rank,i
integer, allocatable,dimension(:) :: chunks

startfun = 1

do i = 1,rank
   startfun = startfun + chunks(i)
end do

end function


end MODULE mc_mod
