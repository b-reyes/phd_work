MODULE twod_ac_mod
USE mpi
use mkl_pardiso

integer :: L_rand,t_pard
double precision :: T_val,a,b,e,f,ggg,hhh,o,p,x_start,x_end 
double precision :: hf,dt
double precision, allocatable, dimension(:) :: omega_xe,omega_ye,par_pie,Phi_x,Phi_y

contains 
!creates the A matrix 
subroutine a_matrix(Aval,Arow,Acol,x)
implicit none 
!declaring variables        
double precision, allocatable, dimension(:),intent(in) :: x
double precision, allocatable, dimension(:),intent(out) :: Aval
integer, allocatable, dimension(:),intent(out) :: Arow,Acol
integer :: m,i
double precision :: h,A1,A2

!getting the size of the x vector
m = size(x)

ALLOCATE(Aval(3*m - 2),Arow(3*m - 2),Acol(3*m - 2))

!finding h value 
h = x(2) - x(1)

A1 = (dble(1)/h)**2*(x(2)-x(2-1)) + (dble(1)/h)**2*(x(2+1) - x(2))
A2 = -(dble(1)/h)**2*(x(2+1) - x(2))

!filling in the middle values                                                                                                         
do i = 1,m-2
   Aval(3*i:2 + 3*i) = (/A2,A1,A2/)
end do

!first two entries                                                                                                                    
Aval(1) = (dble(1)/h)**2*(x(m) - x(m-1))
Aval(2) = -(dble(1)/h)**2*(x(2) - x(1))
!last entries                                                                                                                         
Aval(3*m-2) = (dble(1)/h)**2*(x(m) - x(m-1))
Aval(3*m-3) = -(dble(1)/h)**2*(x(m) - x(m-1))

!middle entries                                                                                                                       
do i = 1,m-2
   Arow(3*i:2 + 3*i) = (/i+1,i+1,i+1/)
end do
!first two indicies                                                                                                                   
Arow(1)= 1
Arow(2) = 1
!last two entries                                                                                                                     
Arow(3*m-2) = m
Arow(3*m-3) = m

!middle entries                                                                                                                       
do i = 1,m-2
   Acol(3*i:2 + 3*i) = (/i,i+1,i+2/)
end do
!first two indicies                                                                                                                   
Acol(1)= 1
Acol(2) = 2
!last two entries                                                                                                                     
Acol(3*m-2) = m
Acol(3*m-3) = m-1

Acol = Acol - 1
Arow = Arow - 1
 
end subroutine a_matrix

!creates the B matrix 
subroutine b_matrix(Bval,Brow,Bcol,x)
implicit none 
!declaring variables                                                                  
double precision, allocatable, dimension(:),intent(in) :: x
double precision, allocatable, dimension(:),intent(inout) :: Bval
integer, allocatable, dimension(:),intent(inout) :: Brow,Bcol
integer :: m,i
double precision :: h,B1,B2

!getting the size of the x vector                                                                                              
m = size(x)

ALLOCATE(Bval(3*m - 2),Brow(3*m - 2),Bcol(3*m - 2))

!finding h value                                                                                                                      
h = x(2) - x(1)

!main diagonal                                                                                                                        
B1 = (dble(1)/h**2)*(-x(2-1)**3/dble(3) + x(2-1)**2*x(2)- x(2-1)*x(2)**2 + x(2)**2*x(2+1) - x(2)*x(2+1)**2 + x(2+1)**3/dble(3))
!sub/super diagonal                                                                                                                   
B2 = (dble(-1)/h**2)*(x(2)**3/dble(6) - x(2)**2*x(2+1)/dble(2) + x(2)*x(2+1)**2/dble(2) - x(2+1)**3/dble(6))

!filling in the middle values 
do i = 1,m-2
   Bval(3*i:2 + 3*i) = (/B2,B1,B2/)
end do 
!first two entries                                                                                                                    
Bval(1) = (dble(1)/h)**2*(x(2)**3/dble(3) - x(1)**3/dble(3) + x(1)**2*x(2) - x(2)**2*x(1))
Bval(2) = (dble(1)/h)**2*(x(2)**3/dble(6) - x(1)*x(2)**2/dble(2) + x(1)**2*x(2)/dble(2) - x(1)**3/dble(6))
!last entries                                                                                                                         
Bval(3*m-2) = (dble(1)/h)**2*(x(m)**3/dble(3) - x(m-1)*x(m)**2 + x(m-1)**2*x(m) - x(m-1)**3/dble(3))
Bval(3*m-3) = (dble(1)/h)**2*(x(m)**3/dble(6) - x(m-1)*x(m)**2/dble(2) + x(m)*x(m-1)**2/dble(2) - x(m-1)**3/dble(6))

!middle entries 
do i = 1,m-2
   Brow(3*i:2 + 3*i) = (/i+1,i+1,i+1/)
end do
!first two indicies 
Brow(1) = 1
Brow(2) = 1
!last two entries 
Brow(3*m-2) = m
Brow(3*m-3) = m

!middle entries                                                                                                                       
do i = 1,m-2
   Bcol(3*i:2 + 3*i) = (/i,i+1,i+2/)
end do
!first two indicies                                                                                                                   
Bcol(1)= 1
Bcol(2) = 2
!last two entries                                                                                                                     
Bcol(3*m-2) = m
Bcol(3*m-3) = m-1

Bcol = Bcol - 1
Brow = Brow - 1

end subroutine b_matrix
       
subroutine CN_FEM(omega_x,omega_y,par_pi,dt,U,eps,eta,center_1,rho)
implicit none 
!declaring variables 
double precision, intent(in) :: dt,eps
double precision, allocatable, dimension(:),intent(in) :: omega_x,omega_y,par_pi
double precision, allocatable, dimension(:),intent(in) :: eta,center_1,rho
double precision, allocatable, dimension(:,:), intent(inout) :: U
double precision, allocatable, dimension(:) :: Y,Uk0
integer :: m,n,z_len,i,j,info
integer :: p
double precision, allocatable, dimension(:) :: Bval
integer, allocatable, dimension(:) :: Brow,Bcol
double precision, allocatable, dimension(:,:) :: B1val
integer, allocatable, dimension(:,:) :: B1row,B1col
double precision, allocatable, dimension(:) :: Aval
integer, allocatable, dimension(:) :: Arow,Acol
double precision, allocatable, dimension(:,:) :: A1_matval
integer, allocatable, dimension(:,:) :: A1_matrow,A1_matcol
double precision, allocatable, dimension(:,:) :: A2_matval
integer, allocatable, dimension(:,:) :: A2_matrow,A2_matcol
double precision, allocatable, dimension(:) :: A1val
integer, allocatable, dimension(:) :: A1row,A1col
double precision, allocatable, dimension(:) :: A2val
integer, allocatable, dimension(:) :: A2row,A2col
integer :: N_ex,job(8),annz,bnnz
double precision, allocatable, dimension(:) :: it1val
integer, allocatable, dimension(:) :: it1row,it1col
double precision, allocatable, dimension(:) :: it2val
integer, allocatable, dimension(:) :: it2row,it2col,pointerB,pointerE,perm
character :: matdescra(6)
integer :: maxfct,mnum,mtype,phase,nrhs,iparm(64),msglvl,error 
TYPE(MKL_PARDISO_HANDLE) :: pt(64)

!dimension on matrices                                                         
N_ex = size(omega_y)

!A_x
call a_matrix(Aval,Arow,Acol,omega_x)

!B_y
call b_matrix(Bval,Brow,Bcol,omega_y)

!number of nonzeros in A_x
annz = size(Aval)

!number of nonzeros in B_y
bnnz = size(Bval)

!kron(A_x,B_y)
call kron(Aval,Arow,Acol,Bval,Brow,Bcol,N_ex,N_ex,annz,bnnz,A1_matval,A1_matrow,A1_matcol)

!dimension of A1_mat
N_ex = size(omega_x)*size(omega_y)

!number of nonzeros in A1_mat
annz = size(A1_matval)

!setting job for conversion from coo to csr                                                                                         
job(1) = 1
job(2) = 1
job(3) = 0
job(6) = 0
job(5) = annz

ALLOCATE(A1row(N_ex+1))
Allocate(A1val(annz),A1col(annz))

!converting to csr format                                                                                                            
call mkl_dcsrcoo(job,N_ex,A1val,A1col,A1row,annz,A1_matval,A1_matrow,A1_matcol,info)

DEALLOCATE(A1_matval,A1_matrow,A1_matcol)

!A_y
call a_matrix(Aval,Arow,Acol,omega_y)
!B_x
call b_matrix(Bval,Brow,Bcol,omega_x)

!dimension on matrices 
N_ex = size(omega_y)

!number of nonzeros in A_y                                                                                                           
annz = size(Aval)

!number of nonzeros in B_x                                                                                                           
bnnz = size(Bval)

!kron(B_x,A_y)
call kron(Bval,Brow,Bcol,Aval,Arow,Acol,N_ex,N_ex,annz,bnnz,A2_matval,A2_matrow,A2_matcol)

!dimension of A2_mat                                                                                                                  
N_ex = size(omega_x)*size(omega_y)

!number of nonzeros in A2_mat                                                                                                         
annz = size(A2_matval)

!setting job for conversion from coo to csr                                                                                           
job(5) = annz

ALLOCATE(A2row(N_ex+1))
Allocate(A2val(annz),A2col(annz))
!converting to csr format                                                                                                             
call mkl_dcsrcoo(job,N_ex,A2val,A2col,A2row,annz,A2_matval,A2_matrow,A2_matcol,info)
DEALLOCATE(A2_matval,A2_matrow,A2_matcol)


ALLOCATE(Aval(annz),Acol(annz),Arow(N_ex + 1))
call mkl_dcsradd('N',0,0,N_ex,N_ex,A1val,A1col,A1row,dble(1),A2val,A2col,A2row,Aval,Acol,Arow,annz,info)

DEALLOCATE(A1val,A1col,A1row)
DEALLOCATE(A2val,A2col,A2row)

!dimension of matrices                                                                                                                
N_ex = size(omega_y)

!B_x                                                                                                                                  
call b_matrix(A1val,A1row,A1col,omega_x)

!B_y                                                                                                                                 
call b_matrix(A2val,A2row,A2col,omega_y)

!number of nonzeros in B_x                                                                                                           
annz = size(A1val)

!number of nonzeros in B_y                                                                                                        
bnnz = size(A2val)

!kron(B_x,B_y)                                                                                                                 
call kron(A1val,A1row,A1col,A2val,A2row,A2col,N_ex,N_ex,annz,bnnz,B1val,B1row,B1col)

!dimension of A1_mat                                                                                                                 
N_ex = size(omega_y)*size(omega_x)

!number of nonzeros in A1_mat                                                                                                   
annz = size(B1val)

!setting job for conversion from coo to csr                                                                                         
job(5) = annz

ALLOCATE(Brow(N_ex+1))
Allocate(Bval(annz),Bcol(annz))

!converting to csr format                                                                                                             
call mkl_dcsrcoo(job,N_ex,Bval,Bcol,Brow,annz,B1val,B1row,B1col,info)

DEALLOCATE(B1val,B1row,B1col)

ALLOCATE(it1val(annz),it1col(annz),it1row(N_ex + 1))
call mkl_dcsradd('N',0,0,N_ex,N_ex,Bval,Bcol,Brow,dt/dble(2),Aval,Acol,Arow,it1val,it1col,it1row,annz,info)

ALLOCATE(it2val(annz),it2col(annz),it2row(N_ex + 1))
call mkl_dcsradd('N',0,0,N_ex,N_ex,Bval,Bcol,Brow,dble(-dt)/dble(2),Aval,Acol,Arow,it2val,it2col,it2row,annz,info)

DEALLOCATE(Aval,Acol,Arow)
DEALLOCATE(Bval,Bcol,Brow)

!allocating the memory for U 
m = size(omega_x)
n = size(omega_y)
z_len = size(par_pi)
ALLOCATE(U(m,n))
U = dble(0)

!building U at t^0
do i = 1,m
   do j = 1,n
      U(i,j) = c_rand_g(omega_x(i),omega_y(j),eps,eta,center_1,rho)
   end do
end do 

!allocating variables to be used in the following loop
ALLOCATE(Uk0(m*n))
Uk0 = reshape(U(:,:),(/m*n/))

ALLOCATE(Y(m*n))

!for matrix-vector multiplication
matdescra(1) = 'S'
matdescra(2) = 'L'
matdescra(3) = 'N'
matdescra(4) = 'F'

ALLOCATE(pointerE(size(it2row)-1),pointerB(size(it2row)-1))

pointerb = it2row(1:N_ex)

pointerE(1:N_ex-1) = it2row(2:N_ex)
pointerE(N_ex) = it2row(N_ex) + 4

DEALLOCATE(it2row)

do i = 1, 64
PT(i)%DUMMY = 0
end do

maxfct = 1
mnum = 1
mtype = 1
phase = 12
nrhs = 1

ALLOCATE(perm(64))

perm = 0

iparm = 0
      
iparm(1) = 1 ! no solver default
iparm(2) = 3 ! fill-in reordering from METIS
iparm(4) = 0 ! no iterative-direct algorithm
iparm(5) = 0 ! no user fill-in reducing permutation
iparm(6) = 0 ! =0 solution on the first n compoments of x
iparm(8) = 0 ! numbers of iterative refinement steps
iparm(10) = 0 ! perturbe the pivot elements with 1E-13
iparm(11) = 0 ! use nonsymmetric permutation and scaling MPS
iparm(13) = 0 ! maximum weighted matching algorithm is switched-off (default for symmetric). Try iparm(13) = 1 in case of inappropriae accuracy
iparm(14) = 0 ! Output: number of perturbed pivots
iparm(18) = -1 ! Output: number of nonzeros in the factor LU
iparm(19) = -1 ! Output: Mflops for LU factorization
iparm(20) = 0 ! Output: Numbers of CG Iterations
iparm(27) = 0 ! check the integer arrays ia and ja

error = 0

msglvl = 0

call mkl_set_dynamic(0)
call mkl_set_num_threads(t_pard)

call pardiso_d(pt, maxfct, mnum, mtype, phase,N_ex,it1val,it1row,it1col,perm,nrhs,iparm,msglvl,Y,Uk0,error)

phase = 33

iparm(8) = 0

error = 0

msglvl = 0

!creating the solution for all of the times starting with the t^1
do p = 2,z_len
   Y = dble(0)
   call mkl_dcsrmv('N',N_ex,N_ex,dble(1),matdescra,it2val,it2col,pointerb,pointere,Uk0,dble(0),Y)   
   call pardiso_d(pt, maxfct, mnum, mtype, phase,N_ex,it1val,it1row,it1col,perm,nrhs,iparm,msglvl,Y,Uk0,error)   
   Uk0 = Uk0/sqrt(exp(-dble(2)*dt/eps**2) + (Uk0**2)*(dble(1) - exp(-dble(2)*dt/eps**2)))
end do

!!!releasing the memory used for Pardiso
phase = -1

call pardiso_d(pt, maxfct, mnum, mtype, phase,N_ex,it1val,it1row,it1col,perm,nrhs,iparm,msglvl,Y,Uk0,error)

DEALLOCATE(it2val,it2col,pointerb,pointere,perm,Y,it1val,it1col,it1row)

U(1:m,1:n) = reshape(Uk0,(/m,n/))

DEALLOCATE(Uk0)

end subroutine CN_FEM
         
subroutine CN_FEM_g(omega_x,omega_y,par_pi,dt,U,eps,eta,center_1,rho)
implicit none
!declaring variables                                       
double precision, intent(in) :: dt,eps
double precision, allocatable, dimension(:),intent(in) :: omega_x,omega_y,par_pi
double precision, allocatable, dimension(:),intent(in) :: eta,center_1,rho
double precision, allocatable, dimension(:,:,:), intent(inout) :: U
double precision, allocatable, dimension(:) :: Y,Uk0
integer :: m,n,z_len,i,j,info
integer :: p,counter
double precision, allocatable, dimension(:) :: Bval
integer, allocatable, dimension(:) :: Brow,Bcol
double precision, allocatable, dimension(:,:) :: B1val
integer, allocatable, dimension(:,:) :: B1row,B1col
double precision, allocatable, dimension(:) :: Aval
integer, allocatable, dimension(:) :: Arow,Acol
double precision, allocatable, dimension(:,:) :: A1_matval
integer, allocatable, dimension(:,:) :: A1_matrow,A1_matcol
double precision, allocatable, dimension(:,:) :: A2_matval
integer, allocatable, dimension(:,:) :: A2_matrow,A2_matcol
double precision, allocatable, dimension(:) :: A1val
integer, allocatable, dimension(:) :: A1row,A1col
double precision, allocatable, dimension(:) :: A2val
integer, allocatable, dimension(:) :: A2row,A2col
integer :: N_ex,job(8),annz,bnnz
double precision, allocatable, dimension(:) :: it1val
integer, allocatable, dimension(:) :: it1row,it1col
double precision, allocatable, dimension(:) :: it2val
integer, allocatable, dimension(:) :: it2row,it2col,pointerB,pointerE,perm
character :: matdescra(6)
integer :: maxfct,mnum,mtype,phase,nrhs,iparm(64),msglvl,error 
TYPE(MKL_PARDISO_HANDLE) :: pt(64)

!dimension on matrices                                                                                                          
N_ex = size(omega_y)

!A_x                                                                                                                            
call a_matrix(Aval,Arow,Acol,omega_x)

!B_y                                                                                                                              
call b_matrix(Bval,Brow,Bcol,omega_y)

!number of nonzeros in A_x                                                                                                       
annz = size(Aval)

!number of nonzeros in B_y                                                                                                      
bnnz = size(Bval)

!kron(A_x,B_y)                                                                                                                  
call kron(Aval,Arow,Acol,Bval,Brow,Bcol,N_ex,N_ex,annz,bnnz,A1_matval,A1_matrow,A1_matcol)

!dimension of A1_mat                                                                                                              
N_ex = size(omega_x)*size(omega_y)

!number of nonzeros in A1_mat                                                                                                     
annz = size(A1_matval)

!setting job for conversion from coo to csr                                                                                       
job(1) = 1
job(2) = 1
job(3) = 0
job(6) = 0
job(5) = annz

ALLOCATE(A1row(N_ex+1))
Allocate(A1val(annz),A1col(annz))

!converting to csr format                                                                                                            
call mkl_dcsrcoo(job,N_ex,A1val,A1col,A1row,annz,A1_matval,A1_matrow,A1_matcol,info)

DEALLOCATE(A1_matval,A1_matrow,A1_matcol)

!A_y
call a_matrix(Aval,Arow,Acol,omega_y)
!B_x                                                                                                            
call b_matrix(Bval,Brow,Bcol,omega_x)

!dimension on matrices                                                                                          
N_ex = size(omega_y)

!number of nonzeros in A_y                                                                                        
annz = size(Aval)

!number of nonzeros in B_x                                                                                           
bnnz = size(Bval)

!kron(B_x,A_y)                                                                                                               
call kron(Bval,Brow,Bcol,Aval,Arow,Acol,N_ex,N_ex,annz,bnnz,A2_matval,A2_matrow,A2_matcol)

!dimension of A2_mat                                                                                                                  
N_ex = size(omega_x)*size(omega_y)

!number of nonzeros in A2_mat                                                                                                         
annz = size(A2_matval)

!setting job for conversion from coo to csr                                                                                           
job(5) = annz

ALLOCATE(A2row(N_ex+1))
Allocate(A2val(annz),A2col(annz))
!converting to csr format                                                                                                             
call mkl_dcsrcoo(job,N_ex,A2val,A2col,A2row,annz,A2_matval,A2_matrow,A2_matcol,info)
DEALLOCATE(A2_matval,A2_matrow,A2_matcol)


ALLOCATE(Aval(annz),Acol(annz),Arow(N_ex + 1))
call mkl_dcsradd('N',0,0,N_ex,N_ex,A1val,A1col,A1row,dble(1),A2val,A2col,A2row,Aval,Acol,Arow,annz,info)

DEALLOCATE(A1val,A1col,A1row)
DEALLOCATE(A2val,A2col,A2row)

!dimension of matrices                                                                                                                
N_ex = size(omega_y)

!B_x                                                                                                                                  
call b_matrix(A1val,A1row,A1col,omega_x)

!B_y                                                                                                                    
call b_matrix(A2val,A2row,A2col,omega_y)

!number of nonzeros in B_x                                                                                                      
annz = size(A1val)

!number of nonzeros in B_y                                                                                                    
bnnz = size(A2val)

!kron(B_x,B_y)                                                                                                         
call kron(A1val,A1row,A1col,A2val,A2row,A2col,N_ex,N_ex,annz,bnnz,B1val,B1row,B1col)

!dimension of A1_mat                                                                                              
N_ex = size(omega_y)*size(omega_x)

!number of nonzeros in A1_mat                                                                                       
annz = size(B1val)

!setting job for conversion from coo to csr                                                                                
job(5) = annz

ALLOCATE(Brow(N_ex+1))
Allocate(Bval(annz),Bcol(annz))

!converting to csr format                                                                                                             
call mkl_dcsrcoo(job,N_ex,Bval,Bcol,Brow,annz,B1val,B1row,B1col,info)

DEALLOCATE(B1val,B1row,B1col)

ALLOCATE(it1val(annz),it1col(annz),it1row(N_ex + 1))
call mkl_dcsradd('N',0,0,N_ex,N_ex,Bval,Bcol,Brow,dt/dble(2),Aval,Acol,Arow,it1val,it1col,it1row,annz,info)

ALLOCATE(it2val(annz),it2col(annz),it2row(N_ex + 1))
call mkl_dcsradd('N',0,0,N_ex,N_ex,Bval,Bcol,Brow,dble(-dt)/dble(2),Aval,Acol,Arow,it2val,it2col,it2row,annz,info)

DEALLOCATE(Aval,Acol,Arow)
DEALLOCATE(Bval,Bcol,Brow)

!allocating the memory for U                                                                                           
m = size(omega_x)
n = size(omega_y)
z_len = size(par_pi)
ALLOCATE(U(m,n,5))
U = dble(0)

!building U at t^0                                                                                                            
do i = 1,m
   do j = 1,n
      U(i,j,1) = c_rand_g(omega_x(i),omega_y(j),eps,eta,center_1,rho)
   end do
end do

!allocating variables to be used in the following loop                                                                      
ALLOCATE(Uk0(m*n))
Uk0 = reshape(U(:,:,1),(/m*n/))

ALLOCATE(Y(m*n))

!for matrix-vector multiplication                                                                                                  
matdescra(1) = 'S'
matdescra(2) = 'L'
matdescra(3) = 'N'
matdescra(4) = 'F'

ALLOCATE(pointerE(size(it2row)-1),pointerB(size(it2row)-1))

pointerb = it2row(1:N_ex)

pointerE(1:N_ex-1) = it2row(2:N_ex)
pointerE(N_ex) = it2row(N_ex) + 4

DEALLOCATE(it2row)

do i = 1, 64
PT(i)%DUMMY = 0
end do

maxfct = 1
mnum = 1
mtype = 1
phase = 12
nrhs = 1

ALLOCATE(perm(64))

perm = 0

iparm = 0

iparm(1) = 1 ! no solver default                                                                     
iparm(2) = 3 ! fill-in reordering from METIS                                                            
iparm(4) = 0 ! no iterative-direct algorithm                                                                     
iparm(5) = 0 ! no user fill-in reducing permutation                                                                
iparm(6) = 0 ! =0 solution on the first n compoments of x                                                            
iparm(8) = 0 ! numbers of iterative refinement steps                                                              
iparm(10) = 0 ! perturbe the pivot elements with 1E-13                                                            
iparm(11) = 0 ! use nonsymmetric permutation and scaling MPS                                                      
iparm(13) = 0 ! maximum weighted matching algorithm is switched-off (default for symmetric).
iparm(14) = 0 ! Output: number of perturbed pivots                                                                 
iparm(18) = -1 ! Output: number of nonzeros in the factor LU                                                        
iparm(19) = -1 ! Output: Mflops for LU factorization                                                                    
iparm(20) = 0 ! Output: Numbers of CG Iterations                                                                    
iparm(27) = 0 ! check the integer arrays ia and ja                                                                       

error = 0

msglvl = 0

call mkl_set_dynamic(0)
call mkl_set_num_threads(t_pard)

call pardiso_d(pt, maxfct, mnum, mtype, phase,N_ex,it1val,it1row,it1col,perm,nrhs,iparm,msglvl,Y,Uk0,error)

phase = 33

iparm(8) = 0

error = 0

msglvl = 0

counter = 2

!creating the solution for all of the times starting with the t^1                                                
do p = 2,z_len

   Y = dble(0)
   call mkl_dcsrmv('N',N_ex,N_ex,dble(1),matdescra,it2val,it2col,pointerb,pointere,Uk0,dble(0),Y)

   call pardiso_d(pt, maxfct, mnum, mtype, phase,N_ex,it1val,it1row,it1col,perm,nrhs,iparm,msglvl,Y,Uk0,error)

   Uk0 = Uk0/sqrt(exp(-dble(2)*dt/eps**2) + (Uk0**2)*(dble(1) - exp(-dble(2)*dt/eps**2)))

   if((p == floor(dble(z_len)/dble(20))) .OR. (p == floor(dble(z_len)/dble(2))) &
        .OR. (p == floor(dble(3*z_len)/dble(4))))then
      U(:,:,counter) = reshape(Uk0,(/m,n/))
      counter = counter + 1
   end if

end do

DEALLOCATE(it2val,it2col,pointerb,pointere,perm,Y,it1val,it1col,it1row)

U(:,:,5) = reshape(Uk0,(/m,n/))

DEALLOCATE(Uk0)

end subroutine CN_FEM_g

function c_rand_g(omega_x,omega_y,eps,eta,center_1,rho)
implicit none
double precision :: omega_x,omega_y,eps,c_rand_g,theta
double precision, allocatable, dimension(:) :: eta,center_1,rho
integer :: L,i,j
double precision :: m
double precision :: c_not

m = dble(3)
L = size(center_1)
c_rand_g = dble(0)

do i =1,L
    do j = 1,L
        theta = angle_g(omega_x,omega_y)
        c_not = tanh((dble(.125) + dble(rho(i))*cos(eta(1)*theta) - sqrt((omega_x &
         - center_1(j))**2 + (omega_y - center_1(j))**2))/(sqrt(dble(2))*eps))
        c_rand_g = c_rand_g + c_not
    end do
end do

end function c_rand_g

function angle(x,y,i,j)
implicit none 
double precision :: x,y,i,j,angle
double precision :: pi     

!calculating pi                                                                         
pi = dble(4)*atan(dble(1))

  if(x > dble(0.25))then
     angle = atan((y-dble(.25))/(x-dble(.25)))  
  else if(x < dble(.25))then
     angle = pi/(i*j) + atan((y-dble(.25))/(x-dble(.25)))
  else
     angle = (pi/(i*j) + dble(.5)*pi)
  end if 


end function angle

function angle_g(x,y)
implicit none
double precision :: x,y,angle_g
double precision :: pi     

!calculating pi                                                                                                                        
pi = dble(4)*atan(dble(1))

  if(x > dble(0.25))then
     angle_g = atan((y-dble(.25))/(x-dble(.25)))
  else if(x < dble(.25))then
     angle_g = pi + atan((y-dble(.25))/(x-dble(.25)))
  else
     angle_g = (pi + dble(.5)*pi)
  end if


end function angle_g

subroutine kron(vals1,rows1,cols1,vals2,rows2,cols2,brows,bcols,annz,bnnz,valst3,rowst3,colst3)

implicit none

double precision,allocatable, dimension(:),intent(inout) :: vals1,vals2
double precision,allocatable, dimension(:,:),intent(out) :: valst3
double precision,allocatable, dimension(:,:) :: vals3
integer, allocatable, dimension(:),intent(inout) :: rows1,cols1,rows2,cols2
integer, allocatable, dimension(:,:),intent(out) :: rowst3,colst3
integer, allocatable, dimension(:,:) :: rows3,cols3
integer,intent(in) :: brows,bcols,annz,bnnz
integer :: i 

!getting the row indicies                                                          
!expanding the entries of A into Blocks                                             
ALLOCATE(rowst3(annz*bnnz,1))
rowst3 = SPREAD(rows1,1,bnnz)
DEALLOCATE(rows1)
rowst3 = rowst3*brows
ALLOCATE(rows3(bnnz,annz))
!increment block indicies                                                           
rows3 = reshape(rowst3,(/bnnz,annz/))
do i = 1,annz
   rows3(:,i) = rows3(:,i)+rows2
end do
DEALLOCATE(rows2)
rowst3 = reshape(rows3,(/annz*bnnz,1/))
DEALLOCATE(rows3)

!getting the column indicies                                                        
ALLOCATE(colst3(annz*bnnz,1))
colst3 = SPREAD(cols1,1,bnnz)
DEALLOCATE(cols1)
colst3 = colst3*bcols
ALLOCATE(cols3(bnnz,annz))
!increment block indicies                                                          
cols3 = reshape(colst3,(/bnnz,annz/))
do i = 1,annz
   cols3(:,i) = cols3(:,i)+cols2
end do
DEALLOCATE(cols2)
colst3 = reshape(cols3,(/annz*bnnz,1/))
DEALLOCATE(cols3)

!getting the updated non-zero values                                                
ALLOCATE(valst3(annz*bnnz,1))

valst3 = SPREAD(vals1,1,bnnz)
DEALLOCATE(vals1)
ALLOCATE(vals3(bnnz,annz))
vals3 = reshape(valst3,(/bnnz,annz/))
do i = 1,annz
   vals3(:,i) = vals3(:,i)*vals2
end do
DEALLOCATE(vals2)
valst3 = reshape(vals3,(/annz*bnnz,1/))
DEALLOCATE(vals3)

end subroutine kron

end MODULE twod_ac_mod
