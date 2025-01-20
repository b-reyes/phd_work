MODULE test_mlmc_mod
USE mpi
use numerical_mod

contains 

subroutine mlmc_test(N0,s_dim,grids,L)
implicit none
!declaring variables                                
integer :: ierror,my_rank,final_ell,i,start,end,one,n_temp,max_ell = 8
integer,intent(in) :: N0,grids(8),s_dim,L
double precision :: time1,time2,sums1(4),sums2(2)
double precision :: al_mlmc,bet_mlmc,gam_mlmc
double precision, allocatable, dimension(:) :: L_vec,L0_vec
double precision :: del1(6),del2(6),var1(6),var2(6),kur1(6),chk1(6),cost(6),check
double precision, dimension(2):: pa,pb
double precision :: pi,mu,eps_2

CALL MPI_Comm_rank(MPI_COMM_WORLD,my_rank,ierror)


if(my_rank == 0)then
   print *, 'Number of samples per level: ',N0
   print *, 'Stochastic dimension: ',s_dim
   print *, 'Number of grid points array: ',grids
   print *, 'The number of levels ran: ','0','-',L
end if 

CALL MPI_BARRIER(MPI_COMM_WORLD,ierror)

final_ell = max_ell

ALLOCATE(L_vec(final_ell + 1),L0_vec(final_ell + 2))

do i = 1,final_ell + 1
   L_vec(i) = i
end do

do i = 0,final_ell + 1
   L0_vec(i+1) = i
end do
pi = dble(4)*atan(dble(1))


del1 = 0
del2 = 0
var1 = 0
var2 = 0
kur1 = 0
chk1 = 0
cost = 0

call numerical_model_init

do i = 0,L
   if(my_rank == 0)then
      print *, 'Level in progress: ',i
   end if
   n_temp = N0
   CALL MPI_BARRIER(MPI_COMM_WORLD,ierror)
   time1 = MPI_Wtime()
   call sampler(i,n_temp,sums1,sums2,grids,s_dim)
   CALL MPI_BARRIER(MPI_COMM_WORLD,ierror)
   time2 = MPI_Wtime()
   cost(i+1) = time2 - time1
   sums1 = sums1/dble(n_temp)
   sums2 = sums2/dble(n_temp)
   del1(i+1) = sums1(1)
   del2(i+1) = sums2(1)
   var1(i+1) = sums1(2)-sums1(1)**2
   var2(i+1) = sums2(2)-sums2(1)**2
   var2(i+1) = max(var2(i+1),dble(10)**(-12))
   kur1(i+1) = (sums1(4) - dble(4)*sums1(3)*sums1(1) + dble(6)*sums1(2)* &
        sums1(1)**2 - dble(3)*sums1(1)**4)/ (sums1(2)-sums1(1)**2)**2

end do

call numerical_model_fin

eps_2 = dble(1)/sqrt(dble(10000))
mu = eps_2**(-2)*sum(sqrt(cost(1:L+1)*var1(1:L+1)))

if(my_rank == 0)then
   print *, 'Optimal N_l for epsilon = ',eps_2,': ',mu*sqrt(var1(1:L+1)/cost(1:L+1))
   print *, 'cost'
   print *, cost
   print *, 'del1 '
   print *, del1
   print *, 'del2 '
   print *, del2
   print *, 'var1 '
   print *, var1
   print *, 'var2 '
   print *, var2
   print *, 'kur1'
   print *, kur1
   print *, 'chk1'
end if

CALL MPI_BARRIER(MPI_COMM_WORLD,ierror)

do i = 0,L
   if(i == 0)then
      check = dble(0)
   else
      check = abs(del1(i+1) + del2(i)  - del2(i+1))/( dble(3.0)*(sqrt(var1(i+1)) + sqrt(var2(i)) + sqrt(var2(i+1)) )/sqrt(dble(N0)))
   end if
   chk1(i + 1) = check
end do

if(my_rank == 0)then
   print *, chk1
end if

start = max(2,floor(dble(.4)*dble(L)))
end = L + 1

one = 1

!finding alpha                                                                 
pa = polyfit(L0_vec(start:end),log(abs(del1(start:end)))/log(dble(2)),one)
al_mlmc = -pa(2)

!finding beta                                                                  
pb = polyfit(L0_vec(start:end),log(abs(var1(start:end)))/log(dble(2)),one)
bet_mlmc = -pb(2)

!finding gamma                                                              
gam_mlmc = log(cost(L+1)/cost(L))/log(dble(2))

if(my_rank == 0)then
   print *, 'Alpha',al_mlmc
   print *, 'Beta', bet_mlmc
   print *, 'Gamma', gam_mlmc
end if

DEALLOCATE(L_vec,L0_vec)

end subroutine mlmc_test
 
!subroutine which deals with computing each sample               
subroutine sampler(ell,n,final_sum1,final_sum2,grids,s_dim)
implicit none 
!declaring variables                                                 
integer :: ierror,my_rank,num_cores,chunks,master = 0,i,q,r
double precision :: pf,pc
integer, intent(inout) :: n
integer, intent(in) :: ell,s_dim
double precision, intent(inout) :: final_sum1(4),final_sum2(2)
double precision :: sum1(4),sum2(2)
integer, allocatable,dimension(:) :: array_chunks
double precision :: pi
double precision, allocatable,dimension(:) :: pf_vec, pc_vec,temp_p,rs
double precision, allocatable,dimension(:,:) :: random_vars
integer :: n_end1,m_end1,constant
integer, intent(in) :: grids(8) 
          
call mpi_comm_rank(mpi_comm_world,my_rank,ierror) !rank of the cores           
call mpi_comm_size(mpi_comm_world,num_cores,ierror)   !number of core         

allocate(array_chunks(num_cores))    !allocating array to hold chunks in                                                                  

if(my_rank == master)then
   chunks = floor(real(n)/num_cores)  !tells us the amount of chunks for each core                                                        

   r = mod(n,num_cores)                !getting the remainder        

   !filling the array_chunks with chunks                             
   array_chunks(1) = chunks

   do i=2,size(array_chunks)
      array_chunks(i) = chunks
   end do

   !distributing the remainder if one exists, in a load balanced way 
   q = 1
   do while(r /= 0)
      array_chunks(q) = array_chunks(q) + 1
      q = q + 1
      r = r - 1
   end do

   if(n < num_cores)then
      array_chunks = 0
      r = n
      q = 1
      do while(r /= 0)
         array_chunks(q) = array_chunks(q) + 1
         q = q + 1
         r = r - 1
      end do
   end if

end if

call mpi_bcast(array_chunks,num_cores,mpi_integer,master,mpi_comm_world,ierror)

!setting constants                                                   
pi = dble(4)*atan(dble(1))

!sums for each vector                                                
sum1 = 0
sum2 = 0
if(1 <= array_chunks(my_rank + 1))then

allocate(pf_vec(array_chunks(my_rank+1)))

n_end1 = grids(ell + 1)
m_end1 = n_end1

allocate(random_vars(array_chunks(my_rank+1),s_dim))

constant = my_rank

ALLOCATE(rs(s_dim))

!running through the standard monte carlo method                     
do i = 1,array_chunks(my_rank+1)
   pf = 0

   call rand_vec(rs,s_dim,my_rank,dble(0),dble(1.0))

   random_vars(i,:) = rs

   call numerical_model(rs,Pf,N_end1,M_end1)   
   
   pf_vec(i) = pf

end do

allocate(pc_vec(array_chunks(my_rank+1)))

n_end1 = (n_end1-1)/2 + 1
m_end1 = (m_end1-1)/2 + 1

do i = 1,array_chunks(my_rank+1)
   if(ell == 0)then
      pc = 0
   else
      !initializing the output quantity of interest                  
      pc = 0

      rs = random_vars(i,:)

      call numerical_model(rs,Pc,N_end1,M_end1)

   end if

   pc_vec(i) = pc

end do

DEALLOCATE(random_vars,rs)

allocate(temp_p(array_chunks(my_rank+1)))

temp_p = pf_vec - pc_vec

sum2(1) =  sum(Pf_vec)
sum2(2) =  sum(Pf_vec**2)

deallocate(pf_vec,pc_vec)

!obtaining the sum for this problem to be used in mlmc               
sum1(1) =  sum(temp_p)
sum1(2) =  sum(temp_p**2)
sum1(3) =  sum(temp_p**3)
sum1(4) =  sum(temp_p**4)

deallocate(array_chunks,temp_p)

end if

call mpi_barrier(mpi_comm_world,ierror)

call mpi_reduce(sum1,final_sum1,4,mpi_double_precision,mpi_sum,master,mpi_comm_world,ierror)

call mpi_bcast(final_sum1,4,mpi_double_precision,master,mpi_comm_world,ierror)

call MPI_REDUCE(sum2,final_sum2,2,MPI_DOUBLE_PRECISION,MPI_SUM,master,MPI_COMM_WORLD,ierror)

CALL MPI_Bcast(final_sum2,2,MPI_DOUBLE_PRECISION,master,MPI_COMM_WORLD,ierror)

call mpi_barrier(mpi_comm_world,ierror)

end subroutine sampler

function logic2int(a)
  implicit none
  logical, intent(in) :: a
  integer :: logic2int

  if(a)then
    logic2int = 1
  else
    logic2int = 0
  end if

end function logic2int


function polyfit(vx, vy, d)
    implicit none
    integer, intent(in) :: d
    integer, parameter :: dp = selected_real_kind(15, 307)
    double precision, dimension(d+1)              :: polyfit
    double precision, dimension(:), intent(in)    :: vx, vy
    double precision, dimension(:,:), allocatable :: X
    double precision, dimension(:,:), allocatable :: XT
    double precision, dimension(:,:), allocatable :: XTX
    integer :: i, j
    integer     :: n, lda, lwork
    integer :: info
    integer, dimension(:), allocatable :: ipiv
    double precision, dimension(:), allocatable :: work

    n = d+1
    lda = n
    lwork = n

    allocate(ipiv(n))
    allocate(work(lwork))
    allocate(XT(n, size(vx)))
    allocate(X(size(vx), n))
    allocate(XTX(n, n))

    ! prepare the matrix                                            
    do i = 0, d
       do j = 1, size(vx)
          X(j, i+1) = vx(j)**i
       end do
    end do

    XT  = transpose(X)
    XTX = matmul(XT, X)

    ! calls to LAPACK subs DGETRF and DGETRI                         
    call DGETRF(n, n, XTX, lda, ipiv, info)
    if ( info /= 0 ) then
       print *, "problem"
       return
    end if
    call DGETRI(n, XTX, lda, ipiv, work, lwork, info)
    if ( info /= 0 ) then
       print *, "problem"
       return
    end if

    polyfit = matmul( matmul(XTX, XT), vy)

    deallocate(ipiv)
    deallocate(work)
    deallocate(X)
    deallocate(XT)
    deallocate(XTX)

end function

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


end MODULE test_mlmc_mod
