MODULE threed_ac_mod
USE mpi
use mkl_pardiso

integer :: L_rand,t_pard
double precision :: T_val,a,b,e,f,ggg,hhh,o,p,x_start,x_end 
double precision :: hf,dt
double precision, allocatable, dimension(:) :: omega_xe,omega_ye,par_pie,Phi_x,Phi_y,Phi_z,omega_ze

contains 
 
!dinzburg-landau free energy calculation
subroutine ginzburg_landau_fe(eps,x_end,x_start,omega_x,omega_y,omega_z,U,final_qoi)
implicit none

double precision :: total_val,total_valx,total_valy,val_1x
double precision :: val_1y,val_2,val_2x,val_2y,val_1,val_3
double precision :: val_3x,val_3y,val_1z,val_2z,val_3z,total_valz
double precision, intent(in) :: eps,x_end,x_start
double precision, allocatable, dimension(:), intent(in) :: omega_x,omega_y,omega_z
double precision, allocatable, dimension(:,:,:), intent(in) :: U
double precision,intent(inout) :: final_qoi
integer :: i,N_u

N_u = size(omega_x)

total_val = dble(0)
total_valx = dble(0)
total_valy = dble(0)
total_valz = dble(0)

do i = 2,N_u-1

   val_1x = (dble(-1)/(omega_x(i)-omega_x(i-1)))*v_funt(omega_y,omega_z,i,i-1,N_u,U)
   val_2x = (dble(1)/(omega_x(i+1)-omega_x(i)) + dble(1)/(omega_x(i)-omega_x(i-1)))*v_funt(omega_y,omega_z,i,i,N_u,U)
   val_3x = (dble(-1)/(omega_x(i+1)-omega_x(i)))*v_funt(omega_y,omega_z,i,i+1,N_u,U)

   total_valx = total_valx + val_1x + val_2x + val_3x

   val_1y = (dble(-1)/(omega_y(i)-omega_y(i-1)))*v_funt(omega_x,omega_z,i,i-1,N_u,U)
   val_2y = (dble(1)/(omega_y(i+1)-omega_y(i)) + dble(1)/(omega_y(i)-omega_y(i-1)))*v_funt(omega_x,omega_z,i,i,N_u,U)
   val_3y = (dble(-1)/(omega_y(i+1)-omega_y(i)))*v_funt(omega_x,omega_z,i,i+1,N_u,U)

   total_valy = total_valy + val_1y + val_2y + val_3y

   val_1z = (dble(-1)/(omega_z(i)-omega_z(i-1)))*v_funt(omega_x,omega_y,i,i-1,N_u,U)
   val_2z = (dble(1)/(omega_z(i+1)-omega_z(i)) + dble(1)/(omega_z(i)-omega_z(i-1)))*v_funt(omega_x,omega_y,i,i,N_u,U)
   val_3z = (dble(-1)/(omega_z(i+1)-omega_z(i)))*v_funt(omega_x,omega_y,i,i+1,N_u,U)

   total_valz = total_valz + val_1z + val_2z + val_3z

   val_1 = ((omega_x(i)-omega_x(i-1))/dble(6))*v_funt(omega_y,omega_z,i,i-1,N_u,U)
   val_2 = ((omega_x(i+1)-omega_x(i-1))/dble(3))*v_funt(omega_y,omega_z,i,i,N_u,U)
   val_3 = ((omega_x(i+1)-omega_x(i))/dble(6))*v_funt(omega_y,omega_z,i,i+1,N_u,U)

   total_val = total_val + val_1 + val_2 + val_3

end do

final_qoi = (dble(1)/(dble(4)*eps**2))*total_val - (dble(1)/(dble(2)*eps**2))*total_val + (dble(1)/(dble(4)*eps**2))*(x_end - x_start)**3 + dble(0.5)*total_valx + dble(0.5)*total_valy + dble(0.5)*total_valz

end subroutine ginzburg_landau_fe

function v_funt(omega_y,omega_z,i,m,N,U)
implicit none
double precision, allocatable, dimension(:) :: omega_y,omega_z
double precision, allocatable, dimension(:,:,:) :: U
integer,intent(in) :: i,m
integer :: j,k,N
double precision :: v_funt,sum1,sum2,sum3,sum1_tot,sum2_tot,sum3_tot

sum1_tot = dble(0)
sum2_tot = dble(0)
sum3_tot = dble(0)

do j = 2,N-1
   sum1 = dble(0)
   sum2 = dble(0)
   sum3 = dble(0)
   do k = 2,N-1
      sum1 = sum1 + v_fun_sub(U(i,j,k),U(m,j-1,k-1),U(m,j-1,k),U(m,j-1,k+1),omega_z,k)

      sum2 = sum2 + v_fun_sub(U(i,j,k),U(m,j,k-1),U(m,j,k),U(m,j,k+1),omega_z,k)

      sum3 = sum3 + v_fun_sub(U(i,j,k),U(m,j+1,k-1),U(m,j+1,k),U(m,j+1,k+1),omega_z,k)
   end do

   sum1_tot = sum1_tot + ((omega_y(j) - omega_y(j-1))/dble(6))*sum1

   sum2_tot = sum2_tot + ((omega_y(j+1) - omega_y(j-1))/dble(3))*sum2

   sum3_tot = sum3_tot + ((omega_y(j+1) - omega_y(j))/dble(6))*sum3

end do

v_funt = sum1_tot + sum2_tot + sum3_tot

end function v_funt


function v_fun_sub(a_v,b_v,c_v,d_v,omega_z,k)
implicit none

double precision,intent(in) :: a_v,b_v,c_v,d_v
double precision, allocatable, dimension(:) :: omega_z
integer,intent(in) :: k
double precision :: v_fun_sub

v_fun_sub = (a_v*b_v)*((omega_z(k) - omega_z(k-1))/dble(6)) + &
      (a_v*c_v)*((omega_z(k+1) - omega_z(k-1))/dble(3)) + &
      (a_v*d_v)*((omega_z(k+1) - omega_z(k))/dble(3))

end function v_fun_sub


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

subroutine CN_FEM(omega_x,omega_y,omega_z,par_pi,dt,U,eta,gam,gam_til,center_1,center_2)
implicit none
!declaring variables                                                 
double precision, intent(in) :: dt
double precision, allocatable, dimension(:),intent(in) :: omega_x,omega_y,par_pi,omega_z
double precision, allocatable, dimension(:),intent(in) :: eta,gam,gam_til
double precision, allocatable, dimension(:),intent(in) :: center_1,center_2
double precision, allocatable, dimension(:,:,:), intent(inout) :: U
double precision, allocatable, dimension(:) :: Bval,Bvalt
integer, allocatable, dimension(:) :: Brow,Bcol,Browt,Bcolt
double precision, allocatable, dimension(:) :: Aval,Avalt
integer, allocatable, dimension(:) :: Arow,Acol,Arowt,Acolt
double precision, allocatable, dimension(:,:) :: A1_matvalt,A2_matvalt
integer, allocatable, dimension(:,:) :: Crowt,Ccolt,A1_matrowt,A1_matcolt,A2_matrowt,A2_matcolt
double precision, allocatable, dimension(:,:) :: Cvalt
integer, allocatable, dimension(:) :: Crow,Ccol,A1_matrow1,A1_matcol1,A2_matrow1,A2_matcol1
double precision, allocatable, dimension(:) :: Cval,A1_matval1,A2_matval1
integer :: N_ex,job(8),annz,bnnz
double precision, allocatable, dimension(:) :: it1val
integer, allocatable, dimension(:) :: it1row,it1col
double precision, allocatable, dimension(:) :: it2val
integer, allocatable, dimension(:) :: it2row,it2col,pointerB,pointerE,perm
character :: matdescra(6)
integer :: maxfct,mnum,mtype,phase,nrhs,iparm(64),msglvl,error 
TYPE(MKL_PARDISO_HANDLE) :: pt(64)
double precision, allocatable, dimension(:) :: Y,Uk0
integer :: w,m,n,z_len,i,j,info,p,k

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
call kron(Aval,Arow,Acol,Bval,Brow,Bcol,N_ex,N_ex,annz,bnnz,Cvalt,Crowt,Ccolt)

!number of nonzeros in kron(A_x,B_y)                                 
annz = size(Cvalt)

!switching over to a different array to be used in kron              
ALLOCATE(Cval(annz))
Cval = Cvalt(:,1)
DEALLOCATE(Cvalt)
ALLOCATE(Crow(annz))
Crow = Crowt(:,1)
DEALLOCATE(Crowt)
ALLOCATE(Ccol(annz))
Ccol = Ccolt(:,1)
DEALLOCATE(Ccolt)

!B_z                                                                 
call b_matrix(Bval,Brow,Bcol,omega_z)

!getting A1 = kron(kron(A_x,B_y),B_z)                                
call kron(Cval,Crow,Ccol,Bval,Brow,Bcol,N_ex,N_ex,annz,bnnz,A1_matvalt,A1_matrowt,A1_matcolt)

!dimension of A1_mat                                                 
N_ex = size(omega_x)*size(omega_y)*size(omega_z)

!number of nonzeros in A1_mat                                        
annz = size(A1_matvalt)

!setting job for conversion from coo to csr                          
job(1) = 1
job(2) = 1
job(3) = 0
job(6) = 0
job(5) = annz

ALLOCATE(A1_matrow1(N_ex+1))
Allocate(A1_matval1(annz),A1_matcol1(annz))

!converting to csr format                                            
call mkl_dcsrcoo(job,N_ex,A1_matval1,A1_matcol1,A1_matrow1,annz,A1_matvalt,A1_matrowt,A1_matcolt,info)

DEALLOCATE(A1_matvalt,A1_matrowt,A1_matcolt)

!dimension on matrices                                               
N_ex = size(omega_y)

!B_x                                                                 
call b_matrix(Bval,Brow,Bcol,omega_x)

!A_y                                                                 
call a_matrix(Aval,Arow,Acol,omega_y)

!number of nonzeros in A_y                                           
annz = size(Aval)

!number of nonzeros in B_x                                           
bnnz = size(Bval)

!kron(B_x,A_y)                                                       
call kron(Bval,Brow,Bcol,Aval,Arow,Acol,N_ex,N_ex,annz,bnnz,Cvalt,Crowt,Ccolt)

!number of nonzeros in kron(B_x,A_y)                                 
annz = size(Cvalt)

!switching over to a different array to be used in kron              
ALLOCATE(Cval(annz))
Cval = Cvalt(:,1)
DEALLOCATE(Cvalt)
ALLOCATE(Crow(annz))
Crow = Crowt(:,1)
DEALLOCATE(Crowt)
ALLOCATE(Ccol(annz))
Ccol = Ccolt(:,1)
DEALLOCATE(Ccolt)

!B_z                                                                 
call b_matrix(Bval,Brow,Bcol,omega_z)

!getting A2 = kron(kron(B_x,A_y),B_z)                                
call kron(Cval,Crow,Ccol,Bval,Brow,Bcol,N_ex,N_ex,annz,bnnz,A2_matvalt,A2_matrowt,A2_matcolt)

!dimension of A2                                                     
N_ex = size(omega_x)*size(omega_y)*size(omega_z)

!number of nonzeros in A2                                            
annz = size(A2_matvalt)

!setting job for conversion from coo to csr                          
job(1) = 1
job(2) = 1
job(3) = 0
job(6) = 0
job(5) = annz

ALLOCATE(A2_matrow1(N_ex+1))
Allocate(A2_matval1(annz),A2_matcol1(annz))

!converting to csr format                                            
call mkl_dcsrcoo(job,N_ex,A2_matval1,A2_matcol1,A2_matrow1,annz,A2_matvalt,A2_matrowt,A2_matcolt,info)

DEALLOCATE(A2_matvalt,A2_matrowt,A2_matcolt)

!computing A = A1 + A2                                               
ALLOCATE(Aval(annz),Acol(annz),Arow(N_ex + 1))
call mkl_dcsradd('N',0,0,N_ex,N_ex,A1_matval1,A1_matcol1,A1_matrow1,dble(1),A2_matval1,A2_matcol1,A2_matrow1,Aval,Acol,Arow,annz,info)

DEALLOCATE(A1_matval1,A1_matcol1,A1_matrow1,A2_matval1,A2_matcol1,A2_matrow1)

!dimension on matrices                                               
N_ex = size(omega_y)

!B_x                                                                 
call b_matrix(Bvalt,Browt,Bcolt,omega_x)

!B_y                                                                 
call b_matrix(Bval,Brow,Bcol,omega_y)

!number of nonzeros in B_x                                           
annz = size(Bvalt)

!number of nonzeros in B_y                                           
bnnz = size(Bval)

!kron(B_x,B_y)                                                       
call kron(Bvalt,Browt,Bcolt,Bval,Brow,Bcol,N_ex,N_ex,annz,bnnz,Cvalt,Crowt,Ccolt)

!number of nonzeros in kron(B_x,B_y)                                 
annz = size(Cvalt)

!switching over to a different array to be used in kron              
ALLOCATE(Cval(annz))
Cval = Cvalt(:,1)
DEALLOCATE(Cvalt)
ALLOCATE(Crow(annz))
Crow = Crowt(:,1)
DEALLOCATE(Crowt)
ALLOCATE(Ccol(annz))
Ccol = Ccolt(:,1)
DEALLOCATE(Ccolt)

!A_z                                                                 
call a_matrix(Avalt,Arowt,Acolt,omega_z)

!getting A1 = kron(kron(B_x,B_y),A_z)                                
call kron(Cval,Crow,Ccol,Avalt,Arowt,Acolt,N_ex,N_ex,annz,bnnz,A1_matvalt,A1_matrowt,A1_matcolt)

!dimension of A1_mat                                                 
N_ex = size(omega_x)*size(omega_y)*size(omega_z)

!number of nonzeros in A1_mat                                        
annz = size(A1_matvalt)

!setting job for conversion from coo to csr                          
job(1) = 1
job(2) = 1
job(3) = 0
job(6) = 0
job(5) = annz

ALLOCATE(A1_matrow1(N_ex+1))
Allocate(A1_matval1(annz),A1_matcol1(annz))

!converting to csr format                                            
call mkl_dcsrcoo(job,N_ex,A1_matval1,A1_matcol1,A1_matrow1,annz,A1_matvalt,A1_matrowt,A1_matcolt,info)

DEALLOCATE(A1_matvalt,A1_matrowt,A1_matcolt)

! A = A + A1                                                         
call mkl_dcsradd('N',0,0,N_ex,N_ex,Aval,Acol,Arow,dble(1),A1_matval1,A1_matcol1,A1_matrow1,Aval,Acol,Arow,annz,info)

DEALLOCATE(A1_matval1,A1_matrow1,A1_matcol1)

!dimension of matrices                                               
N_ex = size(omega_y)

!B_x                                                                 
call b_matrix(Bvalt,Browt,Bcolt,omega_x)

!B_y                                                                 
call b_matrix(Bval,Brow,Bcol,omega_y)

!number of nonzeros in B_x                                           
annz = size(Bvalt)

!number of nonzeros in B_y                                           
bnnz = size(Bval)

!kron(B_x,B_y)                                                       
call kron(Bvalt,Browt,Bcolt,Bval,Brow,Bcol,N_ex,N_ex,annz,bnnz,Cvalt,Crowt,Ccolt)

!number of nonzeros in kron(B_x,B_y)                                 
annz = size(Cvalt)

!switching over to a different array to be used in kron              
ALLOCATE(Cval(annz))
Cval = Cvalt(:,1)
DEALLOCATE(Cvalt)
ALLOCATE(Crow(annz))
Crow = Crowt(:,1)
DEALLOCATE(Crowt)
ALLOCATE(Ccol(annz))
Ccol = Ccolt(:,1)
DEALLOCATE(Ccolt)

!B_z                                                                 
call b_matrix(Avalt,Arowt,Acolt,omega_z)

!getting A1 = kron(kron(B_x,B_y),B_z)                                
call kron(Cval,Crow,Ccol,Avalt,Arowt,Acolt,N_ex,N_ex,annz,bnnz,A1_matvalt,A1_matrowt,A1_matcolt)

!dimension of A1_mat                                                 
N_ex = size(omega_x)*size(omega_y)*size(omega_z)

!number of nonzeros in A1_mat                                        
annz = size(A1_matvalt)

!setting job for conversion from coo to csr                          
job(1) = 1
job(2) = 1
job(3) = 0
job(6) = 0
job(5) = annz

ALLOCATE(Brow(N_ex+1))
Allocate(Bval(annz),Bcol(annz))

!converting to csr format                                            
call mkl_dcsrcoo(job,N_ex,Bval,Bcol,Brow,annz,A1_matvalt,A1_matrowt,A1_matcolt,info)

DEALLOCATE(A1_matvalt,A1_matrowt,A1_matcolt)

ALLOCATE(it1val(annz),it1col(annz),it1row(N_ex + 1))
call mkl_dcsradd('N',0,0,N_ex,N_ex,Bval,Bcol,Brow,dt/dble(2),Aval,Acol,Arow,it1val,it1col,it1row,annz,info)

ALLOCATE(it2val(annz),it2col(annz),it2row(N_ex + 1))
call mkl_dcsradd('N',0,0,N_ex,N_ex,Bval,Bcol,Brow,dble(-dt)/dble(2),Aval,Acol,Arow,it2val,it2col,it2row,annz,info)

DEALLOCATE(Aval,Acol,Arow)
DEALLOCATE(Bval,Bcol,Brow)

!allocating the memory for U                                         
m = size(omega_x)
n = size(omega_y)
w = size(omega_z)
z_len = size(par_pi)
ALLOCATE(U(m,n,w))
U = dble(0)

!building U at t^0                                                   
do i = 1,m
   do j = 1,n
      do k = 1,w
         U(i,j,k) = c_rand(omega_x(i),omega_y(j),omega_z(k),eta,gam, &
              gam_til,center_1,center_2)
      end do
   end do
end do

!allocating variables to be used in the following loop               
ALLOCATE(Uk0(m*n*w))
Uk0 = reshape(U(:,:,:),(/m*n*w/))

ALLOCATE(Y(m*n*w))

!for matrix-vector multiplication                                    
matdescra(1) = 'S'
matdescra(2) = 'L'
matdescra(3) = 'N'
matdescra(4) = 'F'

ALLOCATE(pointerE(size(it2row)-1),pointerB(size(it2row)-1))

pointerb = it2row(1:N_ex)

pointerE(1:N_ex-1) = it2row(2:N_ex)
pointerE(N_ex) = it2row(N_ex) + 8

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
iparm(13) = 0 ! maximum weighted matching algorithm i                
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

do p = 2,z_len

   Y = dble(0)
   call mkl_dcsrmv('N',N_ex,N_ex,dble(1),matdescra,it2val,it2col,pointerb,pointere,Uk0,dble(0),Y)

   call pardiso_d(pt, maxfct, mnum, mtype, phase,N_ex,it1val,it1row,it1col,perm,nrhs,iparm,msglvl,Y,Uk0,error)

   Uk0 = Uk0/sqrt(exp(-dble(2)*dt/gam(1)**2) + (Uk0**2)*(dble(1) - exp(-dble(2)*dt/gam(1)**2)))

end do
!!!releasing the memory used for Pardiso                             
phase = -1

call pardiso_d(pt, maxfct, mnum, mtype, phase,N_ex,it1val,it1row,it1col,perm,nrhs,iparm,msglvl,Y,Uk0,error)

DEALLOCATE(it2val,it2col,pointerb,pointere,perm,Y,it1val,it1col,it1row)

U(1:m,1:n,1:w) = reshape(Uk0,(/m,n,w/))

DEALLOCATE(Uk0)

end subroutine CN_FEM

function c_rand(omega_x,omega_y,omega_z,eta,gam,gam_til,center_1,center_2)
implicit none
double precision :: omega_x,omega_y,omega_z,c_rand,theta
double precision, allocatable, dimension(:) :: eta,gam,gam_til
double precision, allocatable, dimension(:) :: center_1,center_2
integer :: L,i,j,k
double precision :: m
double precision :: c_not

m = dble(3)
L = size(gam_til)
c_rand = dble(0)

do i =1,L
    do j = 1,L
       do k = 1,L
          theta = angle(omega_x,omega_y)
          c_not = tanh((dble(.125) + gam_til(j)*cos(eta(1)*theta) - &
               sqrt((omega_x - center_1(i))**2 + (omega_y - center_1(i))**2 + &
               (omega_z - center_2(k))**2))/(sqrt(dble(2))*gam(1)))
          c_rand = c_rand + c_not
       end do
    end do
end do

end function c_rand

function angle(x,y)
implicit none
double precision :: x,y,angle
double precision :: pi,c1     

c1 = dble(.25)

!calculating pi                                                      
pi = dble(4)*atan(dble(1))

  if(x > dble(c1))then
     angle = atan((y-dble(c1))/(x-dble(c1)))
  else if(x < dble(c1))then
     angle = pi + atan((y-dble(c1))/(x-dble(c1)))
  else
     angle = (pi + dble(0.5)*pi)
  end if


end function angle

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

end MODULE threed_ac_mod
