MODULE FEM_ROUTINES
USE MPI
USE MKL_PARDISO
contains
subroutine runner(T,grids,L_rand,x_start,x_end,a,b,e,f,ggg,hhh,o,p,shift,t_pard)
implicit none
double precision, allocatable,dimension(:) :: par_pi,omega_x,omega_y,temp_err
double precision :: hc,hf,dt,total_valx,total_valy,val12,val22,val32,val_1,val_2,val_3,val_1x,val_2x,val_3x,val_1y,val_2y,val_3y
integer, intent(in) :: grids(7),L_rand,shift,t_pard
double precision,intent(in) :: T,x_start,x_end,a,b,e,f,ggg,hhh,o,p
double precision, allocatable, dimension(:,:) :: U
double precision, allocatable,dimension(:,:) :: Err
double precision,allocatable,dimension(:,:) :: uex,temp
double precision :: time1,time2,eps,total_val,val1,val2,val3,total_val_ex
integer :: i,iii,j,constant
integer :: count,g,temp1,adder,adder2,comp1
double precision :: final_qoi_f,final_qoi_c,total_valy_ex,total_valx_ex
double precision, allocatable, dimension(:) ::  EOC_max,EOC_2,max_err,err_2
double precision, allocatable,dimension(:) :: eta,eta_til,gam_til,center_1,center_2,gam,rho
integer :: my_rank = 0,N_ex,M_ex,N_u,M_u
double precision, allocatable,dimension(:) :: par_pie,omega_xe,omega_ye
integer :: len
len = size(grids)

ALLOCATE(EOC_max(len-2),EOC_2(len-2),max_err(len-1),err_2(len-1))

N_ex = grids(size(grids))
M_ex = grids(size(grids))

hf = dble(x_end - x_start)/dble(N_ex - 1)

ALLOCATE(omega_xe(N_ex))
do j=0,N_ex - 1
   omega_xe(j+1) =  dble(j)*hf
end do

ALLOCATE(omega_ye(N_ex))
omega_ye = omega_xe

dt = T/dble(M_ex-1)

ALLOCATE(par_pie(M_ex))
do j=0,M_ex - 1
   par_pie(j+1) =  dble(j)*dt
end do

ALLOCATE(eta(1),gam(1))
ALLOCATE(center_1(L_rand),rho(L_rand))

call rand_vec(gam,1,constant,ggg,hhh)
constant = constant + shift
call rand_vec(eta,1,my_rank,a,b)
constant = constant + shift
call rand_vec(center_1,L_rand,constant,e,f)
constant = constant + shift
call rand_vec(rho,L_rand,constant,o,p)
eps = gam(1)
DEALLOCATE(gam)

eta = floor(eta)
print *, 'eta',eta
print *, 'x_start',x_start
print *, 'x_end',x_end
print *, 'T',T
print *, 'eps AC',eps
print *, 'start'
time1 = MPI_Wtime()
call CN_FEM(omega_xe,omega_ye,par_pie,dt,uex,eps,eta,center_1,rho,t_pard)
time2 = MPI_Wtime()
print *, 'exact solution time', time2 - time1
print *, ''

DEALLOCATE(par_pie)

count = 1
do g = 1,len - 1
    !declaring variables                                            
    N_u = grids(g)
    M_u = N_u
    dt = T/dble(M_u-1)
    hc = dble(x_end - x_start)/dble(N_u-1)

    ALLOCATE(omega_x(N_u))
    do j=0,N_u - 1
       omega_x(j+1) =  dble(j)*hc
    end do

    ALLOCATE(omega_y(N_u))
    omega_y = omega_x

    ALLOCATE(par_pi(M_u))
    do j=0,M_u-1
       par_pi(j+1) =  dble(j)*dt
    end do

    time1 = MPI_Wtime()
    call CN_FEM(omega_x,omega_y,par_pi,dt,U,eps,eta,center_1,rho,t_pard)
    time2 = MPI_Wtime()
    print *, time2 - time1

    call ginzburg_landau_fe(eps,x_end,x_start,omega_x,omega_y,U,final_qoi_c)

    comp1 = dble(N_ex-1)/dble(N_u-1)

    temp1 = N_u
    ALLOCATE(temp(temp1,temp1))
    temp = dble(0)
    adder = 1
    do j = 1,temp1
       adder2 = 1
       do i = 1,temp1
          temp(j,i) = uex(adder,adder2)
          adder2 = adder2 + comp1
       end do
       adder = adder + comp1
    end do

    call ginzburg_landau_fe(eps,x_end,x_start,omega_x,omega_y,temp,final_qoi_f)

    DEALLOCATE(temp)

    max_err(count) = abs(final_qoi_f - final_qoi_c)

    print *, 'final_qoi_f',final_qoi_f
    print *, 'final_qoi_c',final_qoi_c

!    err_2(count) = sqrt(hc**2*sum(temp_err**2))

    print *, max_err(count)   !,err_2(count)

    count = count + 1

    DEALLOCATE(U,omega_x,omega_y,par_pi)!err,temp_err,par_pi)
end do

DEALLOCATE(omega_xe,omega_ye)
DEALLOCATE(eta,center_1,rho)

do i=1,count - 2
    EOC_max(i) = log(max_err(i)/max_err(i+1))/log(dble(2))
end do

!do i=1,count - 2
!    EOC_2(i) = log(err_2(i)/err_2(i+1))/log(dble(2))
!end do

print *, ''
print *, 'Max_err',max_err
print *, ''
!print *, 'err_2',err_2
!print *, ''
print *, 'EOC_max', EOC_max
print *, ''
!print *, 'EOC_2',EOC_2
DEALLOCATE(EOC_max,max_err,err_2,EOC_2,uex)

end subroutine runner

subroutine ginzburg_landau_fe(eps,x_end,x_start,omega_x,omega_y,U,final_qoi)
implicit none

double precision :: total_val,total_valx,total_valy,val1,val12,val_1,val_1x
double precision :: val_1y,val2,val22,val_2,val_2x,val_2y,val3,val32,val_3
double precision :: val_3x,val_3y
double precision, intent(in) :: eps,x_end,x_start
double precision, allocatable, dimension(:), intent(in) :: omega_x,omega_y
double precision, allocatable, dimension(:,:), intent(in) :: U
double precision,intent(out) :: final_qoi
integer :: i,j,N_u

N_u = size(omega_x)

total_val = dble(0)
total_valx = dble(0)
total_valy = dble(0)
do i = 2,N_u-1
   val1 = dble(0)
   val12 = dble(0)
   do j = 2,N_u-1
      val1 = val1 + ((U(i,j)*U(i-1,j-1))/dble(6))*(omega_y(j) - omega_y(j-1)) &
           + ((U(i,j)*U(i-1,j))/dble(3))*(omega_y(j+1) - omega_y(j-1)) &
           + ((U(i,j)*U(i-1,j+1))/dble(6))*(omega_y(j+1) - omega_y(j))

      val12 = val12 + ((U(i,j)*U(i-1,j-1))/dble(6))*(omega_x(j) - omega_x(j-1)) &
           + ((U(i,j)*U(i-1,j))/dble(3))*(omega_x(j+1) - omega_x(j-1)) &
           + ((U(i,j)*U(i-1,j+1))/dble(6))*(omega_x(j+1) - omega_x(j))
   end do

   val_1 = ((omega_x(i)-omega_x(i-1))/dble(6))*val1
   val_1x = (dble(-1)/(omega_x(i)-omega_x(i-1)))*val1
   val_1y = (dble(-1)/(omega_y(i)-omega_y(i-1)))*val12

   val2 = dble(0)
   val22 = dble(0)
   do j = 2,N_u-1
      val2 = val2 + ((U(i,j)*U(i,j-1))/dble(6))*(omega_y(j) - omega_y(j-1))&
           + ((U(i,j)*U(i,j))/dble(3))*(omega_y(j+1) - omega_y(j-1)) &
           + ((U(i,j)*U(i,j+1))/dble(6))*(omega_y(j+1) - omega_y(j))

      val22 = val22 + ((U(i,j)*U(i,j-1))/dble(6))*(omega_x(j) - omega_x(j-1))&
           + ((U(i,j)*U(i,j))/dble(3))*(omega_x(j+1) - omega_x(j-1)) &
           + ((U(i,j)*U(i,j+1))/dble(6))*(omega_x(j+1) - omega_x(j))

   end do

   val_2 = ((omega_x(i+1)-omega_x(i-1))/dble(3))*val2
   val_2x = (dble(1)/(omega_x(i+1)-omega_x(i)) + dble(1)/(omega_x(i)-omega_x(i-1)))*val2
   val_2y = (dble(1)/(omega_y(i+1)-omega_y(i)) + dble(1)/(omega_y(i)-omega_y(i-1)))*val22

   val3 = dble(0)
   val32 = dble(0)
   do j = 2,N_u-1
      val3 = val3 + ((U(i,j)*U(i+1,j-1))/dble(6))*(omega_y(j) - omega_y(j-1))&
           + ((U(i,j)*U(i+1,j))/dble(3))*(omega_y(j+1) - omega_y(j-1)) &
           + ((U(i,j)*U(i+1,j+1))/dble(6))*(omega_y(j+1) - omega_y(j))

      val32 = val32 + ((U(i,j)*U(i+1,j-1))/dble(6))*(omega_x(j) - omega_x(j-1))&
           + ((U(i,j)*U(i+1,j))/dble(3))*(omega_x(j+1) - omega_x(j-1)) &
           + ((U(i,j)*U(i+1,j+1))/dble(6))*(omega_x(j+1) - omega_x(j))

   end do

   val_3 = ((omega_x(i+1)-omega_x(i))/dble(6))*val3
   val_3x = (dble(-1)/(omega_x(i+1)-omega_x(i)))*val3
   val_3y = (dble(-1)/(omega_y(i+1)-omega_y(i)))*val32

   total_val = total_val + val_1 + val_2 + val_3
   total_valx = total_valx + val_1x + val_2x + val_3x
   total_valy = total_valy + val_1y + val_2y + val_3y

end do

final_qoi = (dble(1)/(dble(4)*eps**2))*total_val - (dble(1)/(dble(2)*eps**2))*total_val + (dble(1)/(dble(4)*eps**2))*(x_end - x_start)*(x_end - x_start)+ dble(0.5)*total_valx + dble(0.5)*total_valy

end subroutine ginzburg_landau_fe




subroutine grapher(T,grids,L_rand,x_start,x_end,a,b,e,f,ggg,hhh,o,p,shift,t_pard)
implicit none
double precision, allocatable,dimension(:) :: par_pi,omega_x,omega_y,temp_err
double precision :: hc,hf,dt
integer, intent(in) :: grids(7),L_rand,shift,t_pard
double precision,intent(in) :: T,x_start,x_end,a,b,e,f,ggg,hhh,o,p
double precision, allocatable, dimension(:,:) :: U
double precision, allocatable,dimension(:,:) :: Err
double precision,allocatable,dimension(:,:,:) :: uex,temp
double precision :: time1,time2,eps
integer :: i,j,constant
integer :: count,g,temp1,adder,adder2,comp1
double precision, allocatable, dimension(:) ::  EOC_max,EOC_2,max_err,err_2
double precision, allocatable,dimension(:) :: eta,eta_til,gam_til,center_1,center_2,gam,rho,mu
integer :: my_rank = 0,N_ex,M_ex,N_u,M_u
double precision, allocatable,dimension(:) :: par_pie,omega_xe,omega_ye
integer :: len
len = size(grids)
ALLOCATE(EOC_max(len-2),EOC_2(len-2),max_err(len-1),err_2(len-1))

N_ex = grids(size(grids))
M_ex = grids(size(grids))

hf = dble(x_end - x_start)/dble(N_ex - 1)

ALLOCATE(omega_xe(N_ex))
do j=0,N_ex - 1
   omega_xe(j+1) =  dble(j)*hf
end do

ALLOCATE(omega_ye(N_ex))
omega_ye = omega_xe

dt = T/dble(M_ex-1)

ALLOCATE(par_pie(M_ex))
do j=0,M_ex - 1
   par_pie(j+1) =  dble(j)*dt
end do

ALLOCATE(eta(1),gam(1))
ALLOCATE(center_1(L_rand),rho(L_rand))

call rand_vec(gam,1,constant,ggg,hhh)
constant = constant + shift
call rand_vec(eta,1,my_rank,a,b)
constant = constant + shift
call rand_vec(center_1,L_rand,constant,e,f)
constant = constant + shift
call rand_vec(rho,L_rand,constant,o,p)
eps = gam(1)
DEALLOCATE(gam)

eta = floor(eta)
print *, 'eta',eta
print *, 'eps AC',eps
print *, 'start'
time1 = MPI_Wtime()
call CN_FEM_g(omega_xe,omega_ye,par_pie,dt,uex,eps,eta,center_1,rho,t_pard)
time2 = MPI_Wtime()
print *, 'exact solution time', time2 - time1
print *, ''

OPEN(UNIT=12,FILE="u_0.txt")

do j = 1,size(uex(:,1,1))
   WRITE(12,*) uex(j,:,1)
end do

close(12)

OPEN(UNIT=12,FILE="u_1_4.txt", ACTION="write", STATUS="replace")

do j = 1,size(uex(:,1,1))
   WRITE(12,*) uex(j,:,2)
end do

close(12)

OPEN(UNIT=12,FILE="u_1_2.txt", ACTION="write", STATUS="replace")

do j = 1,size(uex(:,1,1))
   WRITE(12,*) uex(j,:,3)
end do

close(12)

OPEN(UNIT=12,FILE="u_3_4.txt", ACTION="write", STATUS="replace")

do j = 1,size(uex(:,1,1))
   WRITE(12,*) uex(j,:,4)
end do

close(12)

OPEN(UNIT=12,FILE="u_end.txt", ACTION="write", STATUS="replace")

do j = 1,size(uex(:,1,1))
   WRITE(12,*) uex(j,:,5)
end do

close(12)

DEALLOCATE(uex,eta,gam,eta_til,gam_til,rho,mu)
DEALLOCATE(center_1,center_2,omega_xe,omega_ye,par_pie)


end subroutine grapher


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

!print *, 'in a hi 0',m

ALLOCATE(Aval(3*m - 2),Arow(3*m - 2),Acol(3*m - 2))

!print *, 'in a hi 1'

!finding h value 
h = x(2) - x(1)

A1 = (dble(1)/h)**2*(x(2)-x(2-1)) + (dble(1)/h)**2*(x(2+1) - x(2))
A2 = -(dble(1)/h)**2*(x(2+1) - x(2))

!print *, 'in a hi 2'

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

!print *, 'in a hi 3'

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

!print *, 'in a hi 4'

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

!print *, 'in a hi 5'

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
       
subroutine CN_FEM(omega_x,omega_y,par_pi,dt,U,eps,eta,center_1,rho,t_pard)
implicit none 
!declaring variables 
double precision, intent(in) :: dt,eps
integer, intent(in) :: t_pard
!double precision, dimension(3), intent(in) :: eta,gam
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

!print *, 'hi hi 0'

!print *, 'omega_x',omega_x

!A_x
call a_matrix(Aval,Arow,Acol,omega_x)

!print *, 'hi hi 0.1'

!B_y
call b_matrix(Bval,Brow,Bcol,omega_y)

!print *, 'hi hi 0.15'

!number of nonzeros in A_x
annz = size(Aval)

!number of nonzeros in B_y
bnnz = size(Bval)

!print *, 'hi hi 0.2'

!kron(A_x,B_y)
call kron(Aval,Arow,Acol,Bval,Brow,Bcol,N_ex,N_ex,annz,bnnz,A1_matval,A1_matrow,A1_matcol)

!print *, 'hi hi 0.25'

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

!print *, 'hi hi 0.5'

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

!print *, 'hi hi 0.75'

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

!print *, 'hi hi 0.8'

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

!print *, 'hi hi 0.9'

!converting to csr format                                                                                                             
call mkl_dcsrcoo(job,N_ex,Bval,Bcol,Brow,annz,B1val,B1row,B1col,info)

DEALLOCATE(B1val,B1row,B1col)

ALLOCATE(it1val(annz),it1col(annz),it1row(N_ex + 1))
call mkl_dcsradd('N',0,0,N_ex,N_ex,Bval,Bcol,Brow,dt/dble(2),Aval,Acol,Arow,it1val,it1col,it1row,annz,info)

ALLOCATE(it2val(annz),it2col(annz),it2row(N_ex + 1))
call mkl_dcsradd('N',0,0,N_ex,N_ex,Bval,Bcol,Brow,dble(-dt)/dble(2),Aval,Acol,Arow,it2val,it2col,it2row,annz,info)

DEALLOCATE(Aval,Acol,Arow)
DEALLOCATE(Bval,Bcol,Brow)
!print *, 'hi hi 1'
!allocating the memory for U 
m = size(omega_x)
n = size(omega_y)
z_len = size(par_pi)
ALLOCATE(U(m,n))
U = dble(0)
!print *, 'hi hi 2'
!building U at t^0
do i = 1,m
   do j = 1,n
      U(i,j) = c_rand_g(omega_x(i),omega_y(j),eps,eta,center_1,rho)
   end do
end do 
!print *, 'hi hi 3'

!allocating variables to be used in the following loop
ALLOCATE(Uk0(m*n))
!print *, 'hi hi 3.2'
Uk0 = reshape(U(:,:),(/m*n/))
!print *, 'hi hi 3.3'
ALLOCATE(Y(m*n))

!for matrix-vector multiplication
matdescra(1) = 'S'
matdescra(2) = 'L'
matdescra(3) = 'N'
matdescra(4) = 'F'
!print *, 'hi hi 3.4'
ALLOCATE(pointerE(size(it2row)-1),pointerB(size(it2row)-1))

pointerb = it2row(1:N_ex)
!print *, 'hi hi 3.5'
pointerE(1:N_ex-1) = it2row(2:N_ex)
pointerE(N_ex) = it2row(N_ex) + 4

DEALLOCATE(it2row)


!print *, 'hi hi 3.6'
do i = 1, 64
PT(i)%DUMMY = 0
end do

maxfct = 1
mnum = 1
mtype = 1
phase = 12
nrhs = 1

ALLOCATE(perm(64))
!print *, 'hi hi 3.7'

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

!print *, 'hi hi 4'

call mkl_set_dynamic(0)
call mkl_set_num_threads(t_pard)

call pardiso_d(pt, maxfct, mnum, mtype, phase,N_ex,it1val,it1row,it1col,perm,nrhs,iparm,msglvl,Y,Uk0,error)

!print *, 'hi hi 5'

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

!print *, 'hi hi 6'

!!!releasing the memory used for Pardiso                                    
phase = -1

call pardiso_d(pt, maxfct, mnum, mtype, phase,N_ex,it1val,it1row,it1col,perm,nrhs,iparm,msglvl,Y,Uk0,error)

DEALLOCATE(it2val,it2col,pointerb,pointere,perm,Y,it1val,it1col,it1row)

U(1:m,1:n) = reshape(Uk0,(/m,n/))

DEALLOCATE(Uk0)

end subroutine CN_FEM
         
subroutine CN_FEM_g(omega_x,omega_y,par_pi,dt,U,eps,eta,center_1,rho,t_pard)
implicit none
!declaring variables                                                                                                            
double precision, intent(in) :: dt,eps
integer, intent(in) :: t_pard
!double precision, dimension(3), intent(in) :: eta,gam                                                                            
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

!!!releasing the memory used for Pardiso                                      
phase = -1

call pardiso_d(pt, maxfct, mnum, mtype, phase,N_ex,it1val,it1row,it1col,perm,nrhs,iparm,msglvl,Y,Uk0,error)

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
double precision :: x,y,i,j,angle,k,l
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





end MODULE fem_routines
