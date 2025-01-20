MODULE mlmc_mod
USE mpi
use numerical_mod

contains 

subroutine mlmc(eps,s_dim,n_pows,grids,l_first,gam_mlmc)
implicit none
!declaring variables                                
double precision, intent(in) :: eps,gam_mlmc
integer,intent(in) :: s_dim,n_pows(8),grids(8),l_first
integer :: ierror,my_rank,l,ell,final_ell,i,info,num_cores,n_vals_vec(8)
double precision :: time1,time2,p,rem
double precision :: al_mlmc,bet_mlmc,al_0,bet_0
integer, allocatable, dimension(:) :: nl,dnl,logicals
integer :: range(3) = (/-2,-1,0 /)
double precision, allocatable, dimension(:) :: ml,vl,temp,temp_ml
double precision, allocatable, dimension(:) :: l_vec,l0_vec,work,cl,ns
double precision, allocatable, dimension(:,:) :: suml
double precision, allocatable, dimension(:,:) :: a,power,temp_mat
double precision :: sums(4),infinity = huge(bet_0)
integer :: jj,n0(8),max_ell = 8
logical, allocatable, dimension(:) :: tmp_log
call mpi_comm_rank(mpi_comm_world,my_rank,ierror) !rank of the core                                                   
call mpi_comm_size(mpi_comm_world,num_cores,ierror)   !number of cores = num_cores                                    

n_vals_vec = n_pows

call mpi_barrier(mpi_comm_world,ierror)

if(my_rank == 0)then
   print *, 'Epsilon used: ',eps
   print *, 'Initial numbers of samples per level: ',n_vals_vec
   print *, 'Number of grid points per level: ',grids
   print *, 'Stochastic dimension: ',s_dim
   print *, 'gam_mlmc: ',gam_mlmc
   print *, 'The initial number of levels: ',l_first
   print *, 'Number of cores used: ',num_cores
   !starting the timer                                              
   time1 = mpi_wtime()
end if

n0 = n_vals_vec

final_ell = max_ell

al_mlmc = 0
al_0 = al_mlmc
bet_mlmc = 0
bet_0 = bet_mlmc

allocate(nl(final_ell + 1),dnl(final_ell + 1))
allocate(suml(2,final_ell + 1))
allocate(ml(final_ell + 1),vl(final_ell + 1))
allocate(temp_ml(final_ell + 1),l_vec(final_ell + 1),l0_vec(final_ell + 2))

do i = 1,final_ell + 1
   l_vec(i) = i
end do

do i = 0,final_ell + 1
   l0_vec(i+1) = i
end do

l = l_first
nl = dble(0)
dnl = int(0)
dnl(1:l+1) = n0(1:l+1)
suml = dble(0)

call numerical_model_init

do while((sum(dnl(1:l+1)) > 0))

   ! update sample sums                                              
   do ell=0,l
      if(dnl(ell+1) > 0)then
         call sampler(ell,dnl(ell+1),sums,grids,s_dim)
         nl(ell+1) = nl(ell+1) + dnl(ell+1)
         suml(1,ell+1) = suml(1,ell+1) + sums(1)
         suml(2,ell+1) = suml(2,ell+1) + sums(2)
      end if

      call mpi_barrier(mpi_comm_world,ierror)
   end do

   ml = dble(0)
   vl = dble(0)
   temp_ml = dble(0)
   ! compute absolute average and variance                           
   do ell =0,l
      ml(ell+1) = abs(suml(1,ell+1)/dble(nl(ell+1)))
      temp_ml(ell+1) = suml(2,ell+1)/dble(nl(ell+1)) - ml(ell+1)**2
      vl(ell+1) = max(dble(0),temp_ml(ell+1))
   end do

   ! fix to cope with possible zero values for ml and vl             
   !(can happen in some applications when there are few samples)     
   do ell = 3,l+1
      ml(ell) = max(ml(ell), dble(0.5)*ml(ell-1)/dble(2)**al_mlmc)
      vl(ell) = max(vl(ell), dble(0.5)*vl(ell-1)/dble(2)**bet_mlmc)
   end do

   ! use linear regression to estimate alpha, beta if not given      
   if(al_0 <= 0)then

      allocate(a(l,2),temp_mat(l,2),power(l,2),temp(l),work((2+l)*16))
      a(:,1) = l_vec(1:l)
      a(:,2) = l_vec(1:l)
      power(:,1) =1
      power(:,2) =0
      temp = log(ml(2:l+1))/log(dble(2))
      temp_mat = a**power
      call dgels('n',l,2,1,temp_mat,l,temp,l,work,(2 + l)*16,info)
      al_mlmc = max(dble(0.5),-temp(1))
      if((-temp(1) /= -temp(1)) .or. (-temp(1) >infinity) &
           .or. (-temp(1) < -infinity))then
           al_mlmc = dble(.5)
           if(my_rank == 0)then
              print *, ''
              print *, 'something is infinity for alpha!!'
              print *, ''
           end if
      end if
      deallocate(a,power,temp,work,temp_mat)
      if(my_rank == 0)then
         print *, 'alpha',al_mlmc
      end if
   end if

   if(bet_0 <= 0) then
      allocate(a(l,2),temp_mat(l,2),power(l,2),temp(l),work((2 + l)*16))
      a(:,1) = l_vec(1:l)
      a(:,2) = l_vec(1:l)
      power(:,1) =1
      power(:,2) =0
      temp = log(vl(2:l+1))/log(dble(2))
      temp_mat = a**power
      call dgels('n',l,2,1,temp_mat,l,temp,l,work,(2 + l)*16,info)
      bet_mlmc = max(dble(0.5),-temp(1))
      if((-temp(1) /= -temp(1)) .or. (-temp(1) >infinity) &
           .or. (-temp(1) < -infinity))then
           bet_mlmc = dble(.5)
           if(my_rank == 0)then
              print *, ''
              print *, 'something is infinity for beta!!'
              print *, ''
           end if
      end if
      if(my_rank == 0)then
         print *, 'beta',bet_mlmc
      end if
      deallocate(a,power,temp,work,temp_mat)
   end if

   call mpi_barrier(mpi_comm_world,ierror)

   ! set optimal number of additional samples                        
   allocate(cl(l+1),ns(l+1))
   cl = dble(2)**(gam_mlmc*l0_vec(1:l+1))
   ns  = ceiling(dble(2)*sqrt(vl(1:l+1)/cl(1:l+1))*sum(sqrt(vl(1:l+1)*cl(1:l+1)))/eps**2)
   do jj = 1,l+1
      dnl(jj) = max(dble(0), ns(jj)-dble(nl(jj)))
   end do

   deallocate(cl,ns)

   ! if (almost) converged, estimate remaining error and decide      
   !whether a new level is required                                  
   allocate(logicals(l+1),tmp_log(l+1))
   tmp_log = dble(dnl(1:l+1)) > dble(0.01)*dble(nl(1:l+1))
   do jj =1,l+1
      logicals(jj) = logic2int(tmp_log(jj))
   end do

   if(sum(logicals) == 0 )then
      rem = maxval(ml(l+1+range)*dble(2)**(al_mlmc*range))/dble(dble(2)**al_mlmc - 1)

      if(rem > eps/sqrt(dble(2)))then
         l       = l+1
         vl(l+1) = vl(l)/dble(2)**(bet_mlmc)
         nl(l+1) = 0
         suml(1:4,l+1) = 0
         allocate(cl(l+1),ns(l+1))
         cl = dble(2)**(gam_mlmc*l0_vec(1:l+1))
         ns  = ceiling(dble(2)*sqrt(vl(1:l+1)/cl)*sum(sqrt(vl(1:l+1)*cl))/eps**2)

         do jj = 1,l+1
            dnl(jj) = max(dble(0), ns(jj)-dble(nl(jj)))
         end do

         deallocate(cl,ns)
      end if
   end if

   deallocate(logicals,tmp_log)
   call mpi_barrier(mpi_comm_world,ierror)
end do

call numerical_model_fin

deallocate(dnl,ml,l_vec,l0_vec)

if(my_rank == 0)then
   i = 1
   do while(nl(i) > 0)
      i = i + 1
   end do

   ! finally, evaluate multilevel estimator                         
   p = sum(suml(1,1:i-1)/dble(nl(1:i-1)))

end if

deallocate(suml,temp_ml)
call mpi_barrier(mpi_comm_world,ierror)

if(my_rank == 0)then
   !printing the elapsed time                                        
   time2 = mpi_wtime()
   print *,'Number of samples used per level: ',nl
   print *,'E[Q]: ', p
   print *,'S_E[Q]: ',sqrt(sum(vl(1:i-1)/dble(nl(1:i-1))))
   print *,'elapsed time in seconds (mult. by num_cores): ',(time2 - time1)*dble(num_cores)
end if

deallocate(vl,nl)

end subroutine mlmc
 
!subroutine which deals with computing each sample               
subroutine sampler(ell,n,final_sum1,grids,s_dim)
implicit none 
!declaring variables                                                 
integer :: ierror,my_rank,num_cores,chunks,master = 0,i,q,r
double precision :: pf,pc
integer, intent(inout) :: n
integer, intent(in) :: ell,s_dim
double precision, intent(inout) :: final_sum1(4)
double precision :: sum1(4)
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


end MODULE mlmc_mod
