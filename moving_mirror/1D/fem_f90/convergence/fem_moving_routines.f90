MODULE fem_moving_routines
use mpi

contains

subroutine runner(T,beta,h_vec,k,L_rand,a,b,powpow)
IMPLICIT NONE
!declaring variables                                                           
double precision :: alpha,acon
double precision, intent(in) :: T,beta,k,a,b
integer, intent(in) :: powpow
integer, intent(in) :: L_rand,h_vec(6)
integer :: N,m_final
complex(8), allocatable, dimension(:,:) :: uexact,U,temp,subtract
double precision, allocatable, dimension(:,:) :: Err
double precision, allocatable, dimension(:) :: eta,rs,a_vec
integer :: comp1
double precision :: max_err(10),Err_msq(10),EOC_max(10),EOC_msq(10)
integer :: i,val1,val2,val1_ex,val2_ex,j,adder
double precision :: temp2i,tempi,time1,time2
integer :: ktil,ntil,my_rank

m_final = h_vec(6)
N = h_vec(6)

my_rank = 0
ALLOCATE(rs(L_rand))
call rand_vec(rs,L_rand,my_rank,a,b)

acon = rs(1) + 1.5d0
alpha = k**2

ALLOCATE(eta(L_rand - 1))
eta = rs(2:L_rand)

!solving using a very small time step and acting as if                             
!this is the exact solution                                                         
time1 = MPI_Wtime()
call fem_mover_2(uexact,T,m_final,N,beta,acon,alpha,k,eta,a_vec,powpow)
time2 = MPI_Wtime()
print *, time2 - time1

val1_ex = size(uexact(1,:))
val2_ex = size(uexact(:,1))


DEALLOCATE(a_vec)

do i=1,size(h_vec)-1

    N = h_vec(i)

    time1 = MPI_Wtime()
    call fem_mover_2(U,T,m_final,N,beta,acon,alpha,k,eta,a_vec,powpow)
    time2 = MPI_Wtime()
    print *, time2 - time1

    !choosing the grid comparison                                                  
    comp1 = h_vec(6)/h_vec(i)

    !finding the error of the problem                                                  
    val1 = size(U(1,:))
    val2 = size(U(:,1))
    ALLOCATE(temp(val2,val1),Err(val2,val1),subtract(val2,val1))

    adder = 1
    do j = 1,val2
       temp(j,:) = uexact(adder,:)
       adder = adder + comp1
    end do

    subtract = temp-U

    DEALLOCATE(U,temp)
    Err = cdabs(subtract)

    DEALLOCATE(subtract)
    max_err(i) = maxval(Err)

    !computing the mean-squared nodal error                                             
    temp2i = dble(0)
    do ktil = 1,val1
       tempi = dble(0)
       do ntil = 1,val2
          tempi = tempi + Err(ntil,ktil)**2
       end do
       temp2i = temp2i + (dble(1)/dble(val2))*sqrt(tempi)
    end do 

    Err_msq(i) = (dble(1)/dble(val1))*temp2i

    DEALLOCATE(Err,a_vec)                                                                    
end do

DEALLOCATE(uexact)

!finding the estimated rate of convergence for the max nodal error                  
EOC_max(1) = dble(0)
do i = 2,size(h_vec)-1
   EOC_max(i) = log(max_err(i-1)/max_err(i))/log(dble(2))
end do

!finding the estimated rate of convergence for the mean squared nodal error         
EOC_msq(1) = dble(0)
do i = 2,size(h_vec)-1
   EOC_msq(i) = log(Err_msq(i-1)/Err_msq(i))/log(dble(2))
end do

print *, 'Maximum Error'
print *, max_err
print *, ''
print *, 'Mean Squared Nodal Error'
print *, Err_msq
print *, ''
print *, 'EOC for Maximum Nodal Error'
print *, EOC_max
print *, ''
print *, 'EOC for Mean Squared Nodal Error'
print *, EOC_msq

end subroutine runner

!creates the A matrix 
subroutine A_matrix(A,x)
!declaring variables                                       
double precision, allocatable, dimension(:),intent(in) :: x
double precision, allocatable, dimension(:) :: A_sup,A_diag,A_sub
double precision, allocatable, dimension(:,:),intent(inout) :: A
integer, allocatable,dimension(:) :: j
integer :: m,i

m = size(x)
m = m - 2

ALLOCATE(A(3,m),A_sup(m),A_diag(m),A_sub(m))

ALLOCATE(j(m))

do i = 2,m+1
   j(i-1) = i
end do

!main diagonal                                                       
A_diag(1:m) =  dble(1)/(x(j) - x(j-1)) + dble(1)/(x(j+1) - x(j))

!super diagonal                                                               
A_sup(2:m) = dble(-1)/(x(j(1:m-1)+1)-x(j(1:m-1)))

!sub diagonal                                                                 
A_sub(1:m) = dble(-1)/(x(j)-x(j-1))

DEALLOCATE(j)

A(1,:) = A_sup
A(2,:) = A_diag
A(3,:) = A_sub


DEALLOCATE(A_sup,A_diag,A_sub)

end subroutine a_matrix

!creates the Q matrix                                                      
subroutine Q_matrix(Q,a_vec,xm,xm1)
!declaring variables                                                        
double precision, allocatable, dimension(:),intent(in) :: a_vec,xm,xm1
double precision, allocatable, dimension(:) :: Q_sup,Q_diag,Q_sub
double precision, allocatable, dimension(:,:),intent(inout) :: Q
integer, allocatable, dimension(:) :: i
integer :: m,j

m = size(a_vec)
m = m-2

ALLOCATE(Q(3,m),Q_sup(m),Q_diag(m),Q_sub(m))

ALLOCATE(i(m))

do j = 2,m+1
   i(j-1) = j
end do
 
!main diagonal                                                              
Q_diag(1:m) = (xm(i-1) - xm1(i-1) - xm(i+1) + xm1(i+1))/dble(6)
 
!super diagonal                                                              
Q_sup(2:m) = (dble(2)*xm(i(1:m-1)) - dble(2)*xm1(i(1:m-1)) + xm(i(1:m-1)+1) - xm1(i(1:m-1)+1))/dble(6)
 
!sub diagonal                                                                
Q_sub(1:m) = (dble(-2)*xm(i) + dble(2)*xm1(i)- xm(i-1) + xm1(i-1))/dble(6)

DEALLOCATE(i)

Q(1,:) = Q_sup
Q(2,:) = Q_diag
Q(3,:) = Q_sub

DEALLOCATE(Q_sup,Q_diag,Q_sub)

end subroutine Q_matrix

!creates the M matrix                                                     
subroutine M_matrix(M,x)
!declaring variables                                                   
double precision, allocatable, dimension(:),intent(in) :: x
double precision, allocatable, dimension(:) :: M_sup,M_diag,M_sub
double precision, allocatable, dimension(:,:),intent(inout) :: M
integer, allocatable,dimension(:) :: j
integer :: m_ind,i

m_ind = size(x)
m_ind = m_ind - 2

ALLOCATE(M(3,m_ind),M_sup(m_ind),M_diag(m_ind),M_sub(m_ind))

ALLOCATE(j(m_ind))

do i = 2,m_ind+1
   j(i-1) = i
end do

!main diagonal                                                                
M_diag(1:m_ind) = (x(j+1) - x(j-1))/dble(3)

!super diagonal                                                         
M_sup(2:m_ind) = (x(j(1:m_ind-1)+1)-x(j(1:m_ind-1)))/dble(6)

!sub diagonal                                                               
M_sub(1:m_ind) = (x(j)-x(j-1))/dble(6)

DEALLOCATE(j)                                                       

M(1,:) = M_sup
M(2,:) = M_diag
M(3,:) = M_sub


DEALLOCATE(M_sup,M_diag,M_sub)

end subroutine m_matrix

!gamma function that determines the right boundary             
subroutine gamma(out1,t,acon)  
double precision, intent(in) :: t
double precision :: acon
double precision, intent(inout) :: out1

out1 = acon*t**2

end subroutine gamma

subroutine fem_mover_2(U,T_val,M_end,N_end,beta,acon,alpha,k,eta,a_vec,powpow)
!declaring variables 
complex(8), allocatable, dimension(:,:), intent(inout) :: U
complex(8), allocatable, dimension(:,:) :: RHS,LHS,temp_mat
complex(8), allocatable, dimension(:) :: Y,temp
integer, intent(in) :: powpow
double precision, allocatable, dimension(:),intent(out) :: a_vec 
double precision, allocatable, dimension(:) :: t
double precision, allocatable, dimension(:),intent(in) :: eta
double precision, allocatable, dimension(:) :: hm1,hm,a,am,am1
double precision, allocatable, dimension(:,:) :: A_mat,M_mat,Q_mat
integer, intent(in) :: M_end,N_end
double precision, intent(in) :: T_val,beta,acon,alpha,k
double precision :: dt,h,out1,const
integer :: i,m,num,info
integer,allocatable, dimension(:) :: n
complex(8) :: eye,neye
integer,allocatable, dimension(:) :: IPIV
double precision :: pi                                                       
double precision, allocatable, dimension(:) :: summer

!calculating pi 
pi = dble(4)*atan(dble(1))

eye = dcmplx(dble(0),dble(1))
neye = dcmplx(dble(0),dble(-1))

!creating t from M_end and T_val
dt = dble(T_val)/dble(M_end)

ALLOCATE(t(M_end + 1))

do i = 0,M_end
   t(i+1) = dble(i)*dt
end do

!allocating a vector for all the spacial steps h
ALLOCATE(hm1(M_end+1),hm(M_end+1))

!U matrix full of our unknown constants
Allocate(U(N_end+1,M_end+1))
U = dble(0)

ALLOCATE(n(N_end+1),a(N_end+1),am(N_end+1),am1(N_end+1),a_vec(N_end+1))

do i = 1,N_end+1
   n(i) = i
end do 

!creating our step size for time 1
call gamma(out1,0.d0,acon)
h = dble(out1 - beta)/dble(N_end)
    
!creating the time dependent spacial domain for 1
a = beta + (dble(n) - dble(1))*h

!finding the sum                                                                  
ALLOCATE(summer(size(a))) 

summer = dble(0)

do i=1,size(eta)                                                  
   summer = summer + (eta(i)/dble(i)**powpow)*sin(dble(i)*pi*(a - dble(beta))/dble(-beta))
end do

!initializing the first entry of U
U(:,1) = (exp(eye*dcmplx(k*a,dble(0))))*dcmplx((a - dble(beta))*(a - out1),dble(0)) + &
     dcmplx(summer,dble(0))

deallocate(summer,a)

do m = 2,M_end+1

    !fixed by ganesh
    !creating our step size for the m-1 time
    call gamma(out1,dble(m-1)*dt,acon)
    hm1(m-1) = (out1 - beta)/dble(N_end)
    
    !creating the time dependent spacial domain for m-1
    am1 = beta + dble(n - 1)*hm1(m-1)

    !creating the m-1/2 time dependent spacial domain
    call gamma(out1,(dble(m) - dble(.5))*dt,acon)
    hm1(m-1) = (out1 - beta)/dble(N_end)

    a_vec = beta + dble(n - 1)*hm1(m-1)
    
    !creating our step size for the mth time 
    call gamma(out1,dble(m)*dt,acon)
    hm(m-1) = (out1 - beta)/dble(N_end)
    
    !creating the time dependent spacial domain for m time 
    am = beta + dble(n - 1)*hm(m-1)    

    !creating the m-1/2 time dependent spacial domain 
!    a_vec = (am + am1)/dble(2)

    !creating the matrices needed for each time
    call a_matrix(A_mat,a_vec)
    call m_matrix(M_mat,a_vec)
    call q_matrix(Q_mat,a_vec,am,am1)

    ALLOCATE(LHS(3,size(A_mat(1,:))))

    !creating the left hand side matrix of the equation  -((alpha*dt)/2)*A
    const = alpha*dt/dble(2)
    
    LHS = eye*dcmplx(M_mat,dble(0)) -(eye/dble(2))*dcmplx(Q_mat,dble(0)) - (dcmplx(const,dble(0)))*dcmplx(A_mat,dble(0)) 
    
    !creating the right hand side of the equation
    ALLOCATE(RHS(3,size(A_mat(1,:))))
    RHS = (eye*dcmplx(M_mat,dble(0)) + (eye/dble(2))*dcmplx(Q_mat,dble(0)) + (dcmplx(const,dble(0)))*dcmplx(A_mat,dble(0)))

    num = size(RHS(1,:))
    ALLOCATE(Y(num))
    allocate(temp(2:N_end))
    temp = U(2:N_end,m-1)
    call zgbmv('N',num,num,1,1,-dcmplx(dble(0),dble(1))**2,RHS,3,temp,1,dcmplx(dble(0),dble(0)),Y,1)
    
    !solving for the next time step
    Allocate(IPIV(num))
    ALLOCATE(temp_mat(4,num))
    temp_mat(2:4,1:num) = LHS(1:3,:) 

    call zgbsv(num,1,1,1,temp_mat,4,IPIV,Y,num,info)
    U(2:N_end,m) = Y
    
    DEALLOCATE(LHS,RHS,Y,temp,temp_mat,IPIV,A_mat,M_mat,Q_mat)
    
end do
DEALLOCATE (hm1,hm,n,am,am1,t)

end subroutine fem_mover_2

!creating a random vector for values between a and b
subroutine rand_vec(r_k,j,k,a,b)
implicit none
!declaring variables                                                          
double precision, intent(in) :: a,b
double precision,allocatable, dimension(:),intent(inout) :: r_k
integer :: size,i,time
integer, intent(in) :: j,k
integer,allocatable,dimension(:) :: seed

size = j+12   !setting the size of the vector of seeds (need atleast 12)

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

end MODULE fem_moving_routines
