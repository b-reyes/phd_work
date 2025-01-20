MODULE fem_routines
USE mpi
use mkl_pardiso
contains

subroutine ginzburg_landau_fe(eps,x_end,x_start,omega_x,omega_y,omega_z,U,final_qoi)
implicit none

double precision :: total_val,total_valx,total_valy,val1,val12,val_1,val_1x
double precision :: val_1y,val2,val22,val_2,val_2x,val_2y,val3,val32,val_3
double precision :: val_3x,val_3y,val_1z,val_2z,val_3z,total_valz
double precision, intent(in) :: eps,x_end,x_start
double precision, allocatable, dimension(:), intent(in) :: omega_x,omega_y,omega_z
double precision, allocatable, dimension(:,:,:), intent(in) :: U
double precision,intent(inout) :: final_qoi
integer :: i,j,N_u

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
!double precision,intent(in) :: a,b,c,d
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


function v_fun_sub(a,b,c,d,omega_z,k)
implicit none

double precision,intent(in) :: a,b,c,d
double precision, allocatable, dimension(:) :: omega_z
integer,intent(in) :: k
double precision :: v_fun_sub

v_fun_sub = (a*b)*((omega_z(k) - omega_z(k-1))/dble(6)) + &
      (a*c)*((omega_z(k+1) - omega_z(k-1))/dble(3)) + &
      (a*d)*((omega_z(k+1) - omega_z(k))/dble(3))

end function v_fun_sub

subroutine runner(T,grids,L_rand,x_start,x_end,a,b,c,d,e,f,ggg,hhh,shift,pthr)
implicit none
double precision, allocatable,dimension(:) :: par_pi,omega_x,omega_y,temp_err,par_pie,omega_z,omega_xe,omega_ye,omega_ze
double precision :: hf,hc,dt
integer, intent(in) :: L_rand,shift,pthr
integer, allocatable,dimension(:), intent(in) :: grids 
double precision,intent(in) :: T,x_start,x_end,a,b,c,d,e,f,ggg,hhh
double precision, allocatable, dimension(:,:,:) :: U
double precision, allocatable,dimension(:,:,:) :: Err,temp
double precision,allocatable,dimension(:,:,:) :: uex
double precision :: time1,time2,final_qoi_c,final_qoi_f
integer :: i,j,k,adder3,L,constant,my_rank
integer :: count,g,temp1,adder,adder2,comp1
double precision, allocatable, dimension(:) ::  EOC_max,EOC_2,max_err,err_2,EOC_qoi_max,max_qoi_err
double precision,allocatable,dimension(:) :: eta,gam,eta_til,gam_til,alpha_til
double precision,allocatable,dimension(:) :: center_1, center_2,center_3
integer :: N_u,M_u,N_ex,M_ex,len

my_rank = 0
len = size(grids)

ALLOCATE(EOC_max(len-2),EOC_2(len-2),max_err(len-1),err_2(len-1))
ALLOCATE(EOC_qoi_max(len-2),max_qoi_err(len-1))
N_ex = grids(size(grids))
M_ex = grids(size(grids))

hf = dble(x_end - x_start)/dble(N_ex - 1)

ALLOCATE(omega_xe(N_ex))
do j=0,N_ex - 1
   omega_xe(j+1) =  dble(j)*hf
end do

ALLOCATE(omega_ye(N_ex),omega_ze(N_ex))
omega_ye = omega_xe
omega_ze = omega_ye

dt = T/dble(M_ex-1)

ALLOCATE(par_pie(M_ex))
do j=0,M_ex - 1
   par_pie(j+1) =  dble(j)*dt
end do

L = L_rand

ALLOCATE(eta(1),gam(1),eta_til(L),gam_til(L),alpha_til(L))
ALLOCATE(center_1(L),center_2(L),center_3(L))

call rand_vec(eta,1,my_rank,a,b)
constant = shift + my_rank
call rand_vec(gam,1,constant,ggg,hhh)
constant = shift + constant
call rand_vec(eta_til,L,constant,c,d)
constant = shift + constant
call rand_vec(alpha_til,L,constant,c,d)
constant = shift + constant
call rand_vec(gam_til,L,constant,c,d)
constant = shift + constant
call rand_vec(center_1,L,constant,e,f)
constant = shift + constant
call rand_vec(center_2,L,constant,e,f)
constant = shift + constant
call rand_vec(center_3,L,constant,e,f)

print *, 'gam_til',gam_til

print *, ''

print *, 'center_1',center_1

print *, ''

eta = floor(eta)

print *, 'eps',gam(1)
print *, 'eta',eta

time1 = MPI_Wtime()
call CN_FEM(omega_xe,omega_ye,omega_ze,par_pie,dt,uex,eta,gam, &
     gam_til,center_1,center_2,pthr)
time2 = MPI_Wtime()
print *, 'elapsed time uex', time2-time1
print *, ''

print *, 'uex(1,1,1)',uex(1,1,1)

DEALLOCATE(omega_ye,par_pie,omega_ze) !,omega_xe             
      
count = 1
do g = 1,len-1

    N_u = grids(g)
    M_u = N_u
    dt = T/dble(M_u-1)
    hc = dble(x_end - x_start)/dble(N_u-1)

    ALLOCATE(omega_x(N_u))
    do j=0,N_u - 1
       omega_x(j+1) =  dble(j)*hc
    end do

    ALLOCATE(omega_y(N_u),omega_z(N_u))
    omega_y = omega_x
    omega_z = omega_y

    ALLOCATE(par_pi(M_u))
    do j=0,M_u-1
       par_pi(j+1) =  dble(j)*dt
    end do

    time1 = MPI_Wtime()
   call CN_FEM(omega_x,omega_y,omega_z,par_pi,dt,U,eta,gam, &
              gam_til,center_1,center_2,pthr)
    time2 = MPI_Wtime()

    print *, 'elapsed time U', time2 - time1

    call ginzburg_landau_fe(gam(1),x_end,x_start,omega_x,omega_y,omega_z,U,final_qoi_c)


    comp1 = dble(N_ex-1)/dble(N_u-1)

    print *, 'U(1,1,1)',U(1,1,1)

    print *, 'comp1',comp1

    temp1 = N_u
    ALLOCATE(temp(temp1,temp1,temp1))
    adder = 1
    do j = 1,temp1
       adder2 = 1
       do i = 1,temp1
          adder3 = 1
          do k = 1,temp1
             temp(j,i,k) = uex(adder,adder2,adder3)
             adder3 = adder3 + comp1
          end do
          adder2 = adder2 + comp1
       end do
       adder = adder + comp1
    end do

    call ginzburg_landau_fe(gam(1),x_end,x_start,omega_x,omega_y,omega_z,temp,final_qoi_f) 

    max_qoi_err(count) = abs(final_qoi_f - final_qoi_c)          
    print *, 'final_qoi_f',final_qoi_f                          
    print *, 'final_qoi_c',final_qoi_c 

    ALLOCATE(err(temp1,temp1,temp1))
    err = abs(temp - U)

    DEALLOCATE(temp)

    ALLOCATE(temp_err(temp1*temp1*temp1))
    temp_err = reshape(err,(/temp1*temp1*temp1/))

    max_err(count) = maxval(temp_err)

    err_2(count) = sqrt(hc**3*sum(temp_err**2))

    print *,'errors max/err_2', max_err(count),err_2(count)
    count = count + 1

    DEALLOCATE(U,omega_x,omega_y,par_pi,err,temp_err,omega_z)
end do

DEALLOCATE(eta,gam,eta_til,gam_til,alpha_til)
DEALLOCATE(center_1,center_2,center_3,uex)

do i=1,count - 2
    EOC_max(i) = log(max_err(i)/max_err(i+1))/log(dble(2))
end do

do i=1,count - 2
    EOC_2(i) = log(err_2(i)/err_2(i+1))/log(dble(2))
end do

do i=1,count - 2                                                   
   EOC_qoi_max(i) = log(max_qoi_err(i)/max_qoi_err(i+1))/log(dble(2))                                                                
end do                                                             

print *, 'max_qoi_err',max_qoi_err                                 
print *, ''                                                        
print *, 'EOC_qoi_max',EOC_qoi_max                                 
print *, ''
print *, 'Max_err',max_err
print *, ''
print *, 'err_2',err_2
print *, ''
print *, 'EOC_max', EOC_max
print *, ''
print *, 'EOC_2',EOC_2
print *, ''

DEALLOCATE(max_err,err_2,EOC_max,EOC_2,EOC_qoi_max,max_qoi_err)

end subroutine runner


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


subroutine CN_FEM(omega_x,omega_y,omega_z,par_pi,dt,U,eta,gam,gam_til,center_1,center_2,pthr)
implicit none
!declaring variables                                                   
double precision, intent(in) :: dt
integer, intent(in) :: pthr
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
call mkl_set_num_threads(pthr)

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
double precision, allocatable, dimension(:) :: eta,gam,eta_til,gam_til
double precision, allocatable, dimension(:) :: center_1,center_2,center_3
double precision, allocatable, dimension(:) :: alpha_til
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

!creating a linspace command similar to Matlab for ease of use                          
subroutine linspace(x,x_start,x_end,x_len)
implicit none
!declaring variables                                                               
double precision, allocatable,dimension(:), intent(out) :: x
double precision,intent(in) :: x_start,x_end
double precision :: dx
integer,intent(in) :: x_len
integer:: i 
allocate(x(x_len))
dx = (x_end - x_start)/dble(x_len - 1)
do i=1,x_len
   x(i) = x_start + dble(i-1)*dx
end do
end subroutine linspace

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

!lgwt function that creates weights and points using Legendre_Gauss quadrature         
subroutine lgwt(N,a,b,x,w)
implicit none 
!This script is for computing definite integrals using Legendre-Gauss                   
!Quadrature. Computes the Legendre-Gauss nodes and weights  on an interval              
![a,b] with truncation order N                                                          
!Suppose you have a continuous function f(x) which is defined on [a,b]                  
!which you can evaluate at any x in [a,b]. Simply evaluate it at all of                 
!the values contained in the x vector to obtain a vector f. Then compute                
!the definite integral using sum(f.*w);                                                 
!Written by Greg von Winckel - 02/25/2004                                               
!declaring variables                                                                    
double precision,intent(in) :: a,b
integer,intent(in) :: N
integer :: N1,N2,k,i,N_new  
double precision, allocatable,dimension(:) :: xu,vec,y,y0,Lp
double precision :: pi,eps
double precision, allocatable,dimension(:,:) :: L
double precision,allocatable,dimension(:),intent(inout) :: x,w
double precision :: x_start,x_end

ALLOCATE(x(N),w(N))

!the distance from 1.0 to the next larger double-precision number                       
eps = 2.220446049250313*10**(-16)

!calculating pi                                                                         
pi = dble(4)*atan(dble(1))

N_new=N-1
N1=N_new+1
N2=N_new+2

x_start = dble(-1)
x_end = dble(1)

call linspace(xu,x_start,x_end,N1)     !there was a transpose here!!                    

ALLOCATE(vec(N_new+1))

do i = 1,N_new+1
   vec(i) = dble(i-1)
end do

ALLOCATE(y(N1))
!Initial guess                                                                          
y = cos((dble(2)*vec+dble(1))*pi/(dble(2)*dble(N_new)+dble(2)))+ &
     (dble(0.27)/dble(N1))*sin(pi*xu*dble(N_new)/dble(N2));    !there was a transpose he\re too                                                                                 

!Legendre-Gauss Vandermonde Matrix                                                      
ALLOCATE(L(N1,N2))

!Derivative of LGVM                                                                     
ALLOCATE(Lp(N1))

!Compute the zeros of the N+1 Legendre Polynomial                                       
!using the recursion relation and the Newton-Raphson method                             
ALLOCATE(y0(N1))
y0=dble(2)

!Iterate until new points are uniformly within epsilon of old points                    
do while(MAXVAL(abs(y-y0))>eps)

    L(:,1)=dble(1)
    !Lp(:,1)=0                                                                          

    L(:,2)= y
    !Lp(:,2)=1                                                                          

    do k = 2,N1
        L(:,k+1)=( (dble(2)*dble(k)-dble(1))*y*L(:,k)-dble(k-1)*L(:,k-1) )/dble(k);
    end do

    Lp =dble(N2)*( L(:,N1)-y*L(:,N2) )/(dble(1)-y**2)
    y0=y
    y=y0-L(:,N2)/Lp

end do

!Linear map from[-1,1] to [a,b]                                                         
x=(a*(dble(1)-y)+b*(dble(1)+y))/dble(2)

!Compute the weights                                                                    
w=(b-a)/((dble(1)-y**2)*Lp**2)*(dble(N2)/dble(N1))**2

DEALLOCATE(xu,vec,y,y0,Lp,L)

end subroutine lgwt

!phi function 
function phi(i,a,x)
implicit none 
!declaring variables 
double precision :: phi,x
integer :: i,N
double precision, allocatable,dimension(:) :: a

N = size(a)

if((i == 1) .OR. (i == N))then
    if(i == 1)then
        if((a(1) <= x) .AND. (x <= a(2)))then
            phi = (a(2) - x)/(a(2) - a(1))
        else
            phi = dble(0)
        end if
    elseif(i == N)then
        if((a(N-1) <= x) .AND. (x <= a(N)))then
            phi = (x - a(N-1))/(a(N) - a(N-1))
        else
            phi = dble(0)
        end if
    end if
else
    if((a(i-1) <= x) .AND. (x <= a(i)))then
        phi = (x - a(i-1))/(a(i) - a(i-1))
    elseif((a(i) <= x) .AND. (x <= a(i+1)))then
        phi = (a(i+1) - x)/(a(i+1) - a(i))
    else         
        phi = dble(0)     
    end if
end if

end function phi

function polyfit(vx, vy, d)
    implicit none
    integer, intent(in)                   :: d
    integer, parameter                    :: dp = selected_real_kind(15, 307)
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

subroutine v_fun(k,v)

  implicit none
  integer,parameter :: bits= selected_int_kind(16)
  integer(KIND=bits),intent(in) :: k
  integer(KIND=bits),intent(out) ::v
  integer(KIND=bits) :: a
  integer(KIND=bits) :: one,two,three,four

  v = k
  !swap odd and even bits                                                                                        
  !v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1);                                                
  a = 1431655765
  a = a + int(2,bits)**(32)*a
  one = ishft(v,int(-1,bits))
  two = iand(one,a)
  three = iand(v,a)
  four = ishft(three,int(1,bits)) !int(1,16)                                                                    
  v = xor(two,four)
  ! swap consecutive pairs                                                                                   
  !v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2);                                                     
  a = 858993459
  a = a + int(2,bits)**(32)*a
  one =ishft(v,int(-2,bits))
  two =iand(one,a)
  three= iand(v,a)
  four = ishft(three,int(2,bits))
  v = xor(two,four)
  ! swap nibbles ...                                                                                       
  !v = ((v >> 4) & 0x0F0F0F0F) | ((v & 0x0F0F0F0F) << 4);                                                         
  a = 252645135
  a = a + int(2,bits)**(32)*a
  one =ishft(v,int(-4,bits))
  two =iand(one,a)
  three= iand(v,a)
  four = ishft(three,int(4,bits))
  v = xor(two,four)
  ! swap bytes                                                                                                    
  !v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8);                                                        
  a = 16711935
  a = a + int(2,bits)**(32)*a
  one =ishft(v,int(-8,bits))
  two =iand(one,a)
  three= iand(v,a)
  four = ishft(three,int(8,bits))
  v = xor(two,four)
  ! swap 2-byte long pairs                                                                                         
  !v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8);                                                           
  a = 65535
  a = a + int(2,bits)**(32)*a
  one =ishft(v,int(-16,bits))
  two =iand(one,a)
  three= iand(v,a)
  four = ishft(three,int(16,bits))
  v = xor(two,four)
  ! swap 4-byte long pairs                                                                                            
  !v = ( v >> 32             ) | ( v               << 32);                                                                     
  two = ishft(v,int(-32,bits))
  four = ishft(v,int(32,bits))
  v = xor(two,four)

end subroutine v_fun

subroutine digitalseq_b2ginitial(n,k,s_max,initmode,n_max,cur,recipd,Csr,maxbit)
implicit none
integer,parameter :: bits= selected_int_kind(16)
integer(KIND=bits),allocatable,dimension(:,:) :: Cs
integer(KIND=bits),allocatable,dimension(:,:),intent(inout) :: n,Csr
integer(KIND=bits),allocatable,dimension(:),intent(inout) :: cur
integer(KIND=bits) :: m,temp
integer(KIND=bits),intent(inout) :: s_max,n_max,initmode,maxbit,k
double precision :: recipd  
integer :: i,j,row1,col1

    row1 = size(n(:,1))
    col1 = size(n(1,:))

    ALLOCATE(Cs(row1,col1))

    Cs = n !when intializing we expect the generating matrices as argument 2 (which is n)                             
    m = size(Cs(1,:))
    s_max = size(Cs(:,1))
    n_max = ishft(int(1,bits),m)
    ALLOCATE(Csr(s_max,m))
    do i=1,s_max
       do j=1,m
          call v_fun(Cs(i,j),temp)
          Csr(i,j) = temp
       end do
    end do
    initmode = 0
    maxbit = 64
    recipd = dble(2)**(-maxbit)
    k = 0
    ALLOCATE(cur(s_max))
    cur = 0

    DEALLOCATE(Cs)

end subroutine digitalseq_b2ginitial

subroutine digitalseq_b2g(s,n,k,s_max,initmode,n_max,cur,recipd,Csr,maxbit,x)
implicit none

integer,parameter :: bits= selected_int_kind(16)
integer(KIND=bits),allocatable,dimension(:,:) :: Cs
integer(KIND=bits),allocatable,dimension(:,:),intent(inout) :: Csr
double precision, allocatable,dimension(:,:) :: x
integer(KIND=bits),allocatable,dimension(:),intent(inout) :: cur
integer(KIND=bits),allocatable,dimension(:) :: tmpcsr
integer(KIND=bits) :: m,temp
integer(KIND=bits),intent(inout) :: s,s_max,n_max,initmode,maxbit,k,n
double precision :: recipd  
integer :: i,j,c,si

if(((k + n) > n_max) .OR. (s > s_max))then
    print *, 'Incorrect dimensions for problem'
end if

ALLOCATE(x(s, n))

if((k == 0) .AND. (initmode == 0))then
    x(:, 1) = 0
    si = 2
    k = k + 1
elseif((k == 0) .AND. (initmode == 1))then
    x(:, 1) = 1
    si = 2
    k = k + 1
else
    si = 1
end if

do i=si,n
    c = 1
    do while(ibits(k,int(c-1,bits),1) == 0)
        c = c + 1
    enddo
    ALLOCATE(tmpcsr(size(cur)))
    tmpcsr = Csr(1:s_max,c)
    do j = 1,size(cur)
       cur(j) = xor(cur(j),tmpcsr(j))
    end do
    x(:, i) = dble(cur(1:s))*dble(recipd)
    k = k + 1
    DEALLOCATE(tmpcsr)
end do


end subroutine digitalseq_b2g

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

end MODULE fem_routines
