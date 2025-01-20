module numerical_mod
!So we can use mpi
use mpi 

!So we can use all of the module 
!associated with the 1D model
use oned_mov_mod

contains 

!this subroutine initializes the numerical model to be ran
!all global variables to be used in the model should be 
!initialized here
subroutine numerical_model_init
implicit none 
integer :: ierror,my_rank

CALL MPI_Comm_rank(MPI_COMM_WORLD,my_rank,ierror) !rank of the core  

if(my_rank == 0)then
   OPEN(UNIT=10,FILE="1d_mov_input.txt")
   READ(10,*) T_val
   READ(10,*) beta
   READ(10,*) k
   READ(10,*) a
   READ(10,*) b
   READ(10,*) powpow
   close(10)

   print *, 'T_val: ',T_val
   print *, 'beta: ',beta
   print *, 'k: ', k
   print *, 'a: ',a
   print *, 'b: ',b
   print *, 'powpow: ',powpow
end if

CALL MPI_BARRIER(MPI_COMM_WORLD,ierror)

CALL MPI_Bcast(T_val,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(beta,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(k,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(a,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(b,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(powpow,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)

alpha = k**2

end subroutine numerical_model_init 

!this subroutine takes care of any deallocation or 
!processes that need to be conducted after MC has ran
subroutine numerical_model_fin
implicit none

 

end subroutine numerical_model_fin

subroutine numerical_model(rs,qoi,N_end,M_end)
implicit none
!declaring variables                    
double precision, allocatable, dimension(:),intent(in) :: rs
double precision,intent(inout) :: qoi
integer,intent(in) :: N_end,M_end
double precision, allocatable, dimension(:) :: a_vec,eta,Phi_x
complex(8), allocatable, dimension(:) :: U
integer :: rs_size,ierror,my_rank
integer :: j

CALL MPI_Comm_rank(MPI_COMM_WORLD,my_rank,ierror) !rank of the core  

rs_size = size(rs)

ALLOCATE(eta(rs_size-1))

!creating the random variables for the initial condition
eta = rs(2:rs_size)*(b - a) + a

!creating the nondeterministic shutter speed 
acon = (rs(1)*(b - a) + a) + dble(1.5)

call fem_mover(U,T_val,M_end,N_end,beta,acon,alpha,k,eta,a_vec,powpow) 

!Creating the basis function phi for a specific a_vec                   
ALLOCATE(Phi_x(N_end))
Phi_x = dble(0)
do j = 2,N_end-1
   Phi_x(j) = a_vec(j)**2/(dble(2)*(a_vec(j)-a_vec(j-1))) + a_vec(j)**2/(dble(2)* &       
   (a_vec(j+1)-a_vec(j))) + a_vec(j-1)**2/(dble(2)*(a_vec(j)-a_vec(j-1))) + &      
   a_vec(j+1)**2/(dble(2)*(a_vec(j+1)-a_vec(j)))-a_vec(j-1)*a_vec(j)/ &
   (a_vec(j)-a_vec(j-1)) - a_vec(j+1)*a_vec(j)/(a_vec(j+1)-a_vec(j))
end do

qoi = dble(0)

!Obtaining the intermediate output QoI                                                        
do j = 2,N_end-1
   qoi = qoi + (cdabs(U(j))**2)*Phi_x(j)
end do

DEALLOCATE(U,eta,a_vec,Phi_x)

end subroutine numerical_model

end module numerical_mod



