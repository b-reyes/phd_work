module numerical_mod
!So we can use mpi
use mpi 

!So we can use all of the module 
!associated with the 2D AC model
use twod_ac_mod

contains 

!this subroutine initializes the numerical model to be ran
!all global variables to be used in the model should be 
!initialized here
subroutine numerical_model_init
implicit none 
integer :: ierror,my_rank

CALL MPI_Comm_rank(MPI_COMM_WORLD,my_rank,ierror) !rank of the core  

if(my_rank == 0)then
   OPEN(UNIT=10,FILE="2d_ac_input.txt")
   READ(10,*) T_val
   READ(10,*) a
   READ(10,*) b
   READ(10,*) e
   READ(10,*) f
   READ(10,*) ggg
   READ(10,*) hhh
   READ(10,*) o
   READ(10,*) p
   READ(10,*) L_rand
   READ(10,*) x_start
   READ(10,*) x_end
   READ(10,*) t_pard
   close(10)

   print *, 'T_val: ',T_val
   print *, 'a: ',a
   print *, 'b: ',b
   print *, 'e: ',e
   print *, 'f: ',f
   print *, 'ggg: ',ggg
   print *, 'hhh: ',hhh
   print *, 'o: ',o
   print *, 'p: ',p
   print *, 'L_rand: ',L_rand
   print *, 'x_start: ',x_start
   print *, 'x_end: ',x_end
   print *, 't_pard: ',t_pard

end if

CALL MPI_BARRIER(MPI_COMM_WORLD,ierror)

CALL MPI_Bcast(T_val,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(a,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(b,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(e,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(f,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(ggg,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(hhh,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(o,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(p,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(L_rand,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(x_start,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(x_end,1,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierror)
CALL MPI_Bcast(t_pard,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)

!N_ex = N_ex_s

!M_ex = M_ex_s

!hf = dble(x_end - x_start)/dble(N_ex - 1)

!ALLOCATE(omega_xe(N_ex))
!do j=0,N_ex - 1
!   omega_xe(j+1) =  dble(j)*hf
!end do

!ALLOCATE(omega_ye(N_ex))
!omega_ye = omega_xe

!dt = T_val/dble(M_ex-1)

!ALLOCATE(par_pie(M_ex))
!do j=0,M_ex - 1
!   par_pie(j+1) =  dble(j)*dt
!end do

!ALLOCATE(Phi_x(N_ex),Phi_y(N_ex))

!computing the integrals of phi_x and phi_y                          
!do i = 2,N_ex-1
!   Phi_x(i) = omega_xe(i-1)**2/(dble(2)*hf) - (omega_xe(i-1)*omega_xe(i))/hf + omega_xe(i)**2/hf - (omega_xe(i)*omega_xe(i+1))/hf &
!        + omega_xe(i+1)**2/(dble(2)*hf)
!end do

!Phi_x(1) = (dble(-1)/hf)*(-omega_xe(2)**2/dble(2) - omega_xe(1)**2/dble(2) + omega_xe(1)*omega_xe(2))
!Phi_x(N_ex) = omega_xe(N_ex-1)**2/(dble(2)*hf) - (omega_xe(N_ex-1)*omega_xe(N_ex))/hf+ omega_xe(N_ex)**2/(dble(2)*hf)

!Phi_y = Phi_x

end subroutine numerical_model_init

!this subroutine takes care of any deallocation or             
!processes that need to be conducted after MC has ran                
subroutine numerical_model_fin
implicit none

!DEALLOCATE(Phi_y,Phi_x,omega_xe,omega_ye,par_pie)

end subroutine numerical_model_fin

subroutine numerical_model(rs,qoi,N_end,M_end)
implicit none
!declaring variables                    
double precision, allocatable, dimension(:),intent(in) :: rs
double precision,intent(out) :: qoi
integer,intent(in) :: N_end,M_end
double precision, allocatable, dimension(:) :: eta,gam,center_1,rho
double precision, allocatable, dimension(:,:) :: U
integer :: rs_size,ierror,my_rank
integer :: j,w,row,col,N_ex,M_ex,i

CALL MPI_Comm_rank(MPI_COMM_WORLD,my_rank,ierror) !rank of the core                         

rs_size = size(rs)

ALLOCATE(eta(1),gam(1))
ALLOCATE(center_1(L_rand),rho(L_rand))

gam(1) = rs(1)*(hhh - ggg) + ggg
eta(1) = floor(rs(2)*(b - a) + a)
center_1 = rs(3:2+L_rand)*(e - f) + f
rho = rs(L_rand:rs_size)*(o - p) + p

N_ex = N_end

M_ex = M_end

hf = dble(x_end - x_start)/dble(N_ex - 1)

ALLOCATE(omega_xe(N_ex))
do j=0,N_ex - 1
   omega_xe(j+1) =  dble(j)*hf
end do

ALLOCATE(omega_ye(N_ex))
omega_ye = omega_xe

dt = T_val/dble(M_ex-1)

ALLOCATE(par_pie(M_ex))
do j=0,M_ex - 1
   par_pie(j+1) =  dble(j)*dt
end do

ALLOCATE(Phi_x(N_ex),Phi_y(N_ex))

!computing the integrals of phi_x and phi_y                          
do i = 2,N_ex-1
   Phi_x(i) = omega_xe(i-1)**2/(dble(2)*hf) - (omega_xe(i-1)*omega_xe(i))/hf + omega_xe(i)**2/hf - (omega_xe(i)*omega_xe(i+1))/hf &
        + omega_xe(i+1)**2/(dble(2)*hf)
end do

Phi_x(1) = (dble(-1)/hf)*(-omega_xe(2)**2/dble(2) - omega_xe(1)**2/dble(2) + omega_xe(1)*omega_xe(2))
Phi_x(N_ex) = omega_xe(N_ex-1)**2/(dble(2)*hf) - (omega_xe(N_ex-1)*omega_xe(N_ex))/hf+ omega_xe(N_ex)**2/(dble(2)*hf)

Phi_y = Phi_x

!runs the deterministic solve of the 2d ac stochastic model
call CN_FEM(omega_xe,omega_ye,par_pie,dt,U,gam(1),eta,center_1,rho)

DEALLOCATE(omega_xe,omega_ye,par_pie)

qoi = dble(0)

row = size(U(:,1))
col = size(U(1,:))

!obtaining the QoI
do w = 1,row
   do j = 1,col
      qoi = U(w,j)*Phi_x(w)*Phi_y(j) + qoi
   end do
end do

DEALLOCATE(U,eta,gam,center_1,rho,Phi_x,Phi_y)

end subroutine numerical_model


end module numerical_mod



