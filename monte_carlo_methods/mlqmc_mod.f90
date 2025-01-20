MODULE mlqmc_mod
USE mpi
use numerical_mod
USE,INTRINSIC :: ISO_C_BINDING
INTERFACE
    Subroutine digitalshift(s,M_iter,j,jj,nprev,nn,r_beg,r_end) BIND(c,NAME='digitalshift')
    use iso_c_binding
    integer(c_int),value :: s,M_iter,jj,j,nprev,nn,r_beg,r_end

    END subroutine digitalshift

    Subroutine shiftgen(s,M_iter) BIND(c,NAME='shiftgen')
    use iso_c_binding
    integer(c_int),value :: s,M_iter

    END subroutine shiftgen

END INTERFACE
contains 

subroutine mlqmc(eps,s_dim,n_pows,grids,l_first,m_it_pow)
use iso_c_binding
implicit none
!declaring variables                                                 
integer, intent(in) :: n_pows(8),grids(8),m_it_pow
!,n_vals_vec(5),l_rand,l_first,max_ell
integer, intent(in) :: s_dim,l_first
double precision, intent(in) :: eps
integer :: ierror,my_rank,master = 0,l,ell,final_ell,i,num_cores
double precision :: time1,time2,time3,time4,rem
double precision :: al_mlmc,bet_mlmc,al_0,bet_0
integer, allocatable, dimension(:) :: nl,dnl,global_nl,range
integer(c_int), allocatable,dimension(:) :: m_iter
double precision, allocatable, dimension(:) :: ml,vl,temp_ml,acc_tot,q,varq
double precision, allocatable, dimension(:) :: l_vec,l0_vec,cl
double precision, allocatable, dimension(:,:) :: suml
double precision, allocatable,dimension(:,:) :: diff_acc
double precision :: sums(4)
integer :: n0(8),n_vals_vec(8)
integer :: flag,flag2,l_star,maxmax,max_ell = 8
integer(c_int), allocatable,dimension(:) :: nn,nprev

call mpi_comm_rank(mpi_comm_world,my_rank,ierror) !rank of the core  
call mpi_comm_size(mpi_comm_world,num_cores,ierror)   !number of core

n_vals_vec = n_pows

call mpi_barrier(mpi_comm_world,ierror)

if(my_rank == master)then
   print *, 'Epsilon used: ',eps
   print *, 'Initial numbers of samples per level: ',n_vals_vec
   print *, 'Number of grid points per level: ',grids
   print *, 'Stochastic dimension: ',s_dim
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
allocate(global_nl(final_ell + 1),cl(final_ell + 1))
allocate(nl(final_ell + 1),dnl(final_ell + 1))
allocate(suml(2,final_ell + 1))
allocate(q(final_ell + 1),m_iter(final_ell + 1))
allocate(ml(final_ell + 1),vl(final_ell + 1))
allocate(temp_ml(final_ell + 1),l_vec(final_ell + 1),l0_vec(final_ell + 2))
allocate(nn(final_ell + 1),nprev(final_ell + 1))

allocate(range(final_ell+1))

do i = final_ell,0
   range(i+1) = -i
end do

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

flag = .true.
flag2 = .true.

diff_acc = dble(0)

nn = int(0,c_int)

nprev = int(0,c_int)

do i = 1,final_ell + 1
   m_iter(i) = int(2,c_int)**m_it_pow
   if((i-1) == 1)then
      m_iter(i) = int(2,c_int)**m_it_pow
   else if((i-1) == 0)then
      m_iter(i) = int(2,c_int)**m_it_pow
   else if((i-1) > 1)then
      m_iter(i) = int(2,c_int)**m_it_pow
   end if
end do

maxmax = maxval(m_iter)

allocate(diff_acc(final_ell + 1,maxmax))

diff_acc = dble(0)

call numerical_model_init

do while(flag == .true.)

   ! update sample sums                                              
   do ell=0,l
      nn(ell + 1) = nprev(ell+1) + dnl(ell + 1)
      if(dnl(ell+1) > 0)then
         call mpi_barrier(mpi_comm_world,ierror)
         time3 = mpi_wtime()
         call qsampler(ell,sums,grids,s_dim,m_iter(ell+1),nn(ell+1),nprev(ell+1),diff_acc)
         call mpi_barrier(mpi_comm_world,ierror)
         time4 = mpi_wtime()
         cl(ell+1) = cl(ell+1) + (time4 - time3)
         global_nl(ell+1) = dnl(ell+1)
         nl(ell+1) = nl(ell+1) + dnl(ell+1)
         suml(1,ell+1) = suml(1,ell+1) + sums(1)
         suml(2,ell+1) = suml(2,ell+1) + sums(2)
      end if
      nprev(ell +1) = nn(ell+1)

      call mpi_barrier(mpi_comm_world,ierror)
   end do

   ml = dble(0)
   vl = dble(0)
   temp_ml = dble(0)
   ! compute absolute average and variance                           
   do ell =0,l
      ml(ell+1) = abs(suml(1,ell+1)/dble(nl(ell+1)*dble(m_iter(ell+1))))
   end do

   allocate(acc_tot(l+1))
   acc_tot = 0 !calculate average over all shifts:              

   do ell = 0,l
      do i = 1,m_iter(ell+1)
         acc_tot(ell+1) = acc_tot(ell+1) + diff_acc(ell+1,i)
      end do
   end do
                  
   do ell = 0,l
      q(ell+1) = acc_tot(ell+1)/dble(nl(ell+1)*m_iter(ell+1))
   end do

   deallocate(acc_tot)

   allocate(varq(l+1))
   varq = dble(0) !calculate variance of estimator q(g)              

   do ell = 0,l
      do i = 1,m_iter(ell+1)
         varq(ell+1) = varq(ell+1) + (diff_acc(ell+1,i)/dble(nl(ell+1)) - q(ell+1))**2
      end do
   end do                  

   do ell = 0,l
      varq(ell+1) = varq(ell+1)/dble(m_iter(ell+1)*dble(m_iter(ell+1) - 1))
   end do


   if(sum(varq(1:l+1)) > dble(.5)*eps**2)then

      l_star = maxloc(varq(1:l+1)/cl(1:l+1),dim=1) - 1
      dnl = 0
      dnl(l_star + 1) = 2*global_nl(l_star + 1)
      if(dnl(l_star + 1) == 0)then
         dnl(l_star + 1) = 1
      end if

   else

      rem = q(l+1)**2

      if((rem > eps/sqrt(dble(2))) .and. (l <= 4))then
         l       = l+1
         dnl = int(0)
         dnl(l+1) = int(1)
      else
         flag = .false.
         flag2 = .false.
         call mpi_barrier(mpi_comm_world,ierror)
      end if
   end if

   if(flag2 == .true.)then
      deallocate(varq)
   end if
end do

call numerical_model_fin

deallocate(dnl,ml,l_vec,l0_vec,cl,nn,nprev,m_iter,diff_acc)

deallocate(suml,temp_ml)
call mpi_barrier(mpi_comm_world,ierror)

if(my_rank == master)then
   !printing the elapsed time                                        
   time2 = mpi_wtime()
   print *, 'Number of samples used per level: ',nl
   print *, 'E[Q]: ', sum(q)
   print *, 'S_E[Q]: ',sqrt(sum(varq))
   print *, 'elapsed time in seconds (mult. by num_cores): ',(time2 - time1)*dble(num_cores)
end if

deallocate(nl,q,varq)

end subroutine mlqmc

!subroutine which deals with computing each sample                  
subroutine qsampler(ell,final_sum1,grids,s_dim,m_iter,nn,nprev,diff_acc)
use iso_c_binding
implicit none 
!declaring variables                                                 
integer :: ierror,my_rank,num_cores,chunks,master = 0,i,qq,rrr
double precision :: Pf,Pc,pi
integer, intent(in) :: ell,s_dim
!double precision, intent(in) :: T,a,b,ggg,hhh,e,f,o,pp,x_end,x_start
double precision, intent(inout) :: final_sum1(4)
double precision, allocatable,dimension(:) :: acc1,acc2,temp_diff,final_acc
double precision, allocatable,dimension(:,:), intent(inout) :: diff_acc
double precision :: sum1(4)
integer status(MPI_STATUS_SIZE),my_comm
integer, allocatable,dimension(:) :: array_chunks,groups,arr_c2
double precision, allocatable, dimension(:,:) :: aa,random_vars2
double precision, allocatable,dimension(:) :: Pf_vec, Pc_vec,temp_P,rs
double precision, allocatable,dimension(:,:,:) :: random_vars,Y,aa2
integer :: N_end1,M_end1,constant,num_cores_used,N_pow,my_new_rank,color,counter,start,ende,counter2
integer, intent(in) :: grids(8)
integer(c_int), intent(inout) :: nn,nprev
integer(c_int),intent(inout) :: m_iter
integer(c_int) :: r,s                                     

CALL MPI_Comm_rank(MPI_COMM_WORLD,my_rank,ierror) !rank of the cores          
CALL MPI_COMM_Size(MPI_COMM_WORLD,num_cores,ierror)   !number of core


s = int(s_dim,c_int) !Dimension                                 
N_pow = 16

if(my_rank == 0)then

   !creating the shifts and writing them to a file                   
   call shiftgen(s_dim,M_iter)


   if(num_cores > M_iter)then
      ALLOCATE(array_chunks(M_iter))    !allocating array to hold chunks in                                                               
      chunks = floor(dble(num_cores)/dble(M_iter))
      rrr = mod(num_cores,M_iter)
      array_chunks = chunks

      !distributing the remainder if one exists, in a load balanced way                                                                   
      qq = 1
      do while(rrr /= 0)
         array_chunks(qq) = array_chunks(qq) + 1
         qq = qq + 1
         rrr = rrr - 1
      end do

      allocate(groups(M_iter))

      groups(1) = array_chunks(1)-1
      do i =2,M_iter
         groups(i) = groups(i-1) + array_chunks(i)
      end do

      ALLOCATE(Y(nn - nprev,s_dim,M_iter))

      do r = 1,M_iter
         !applying a digital shift to the sobol points          
         call digitalshift(s_dim,M_iter,N_pow,N_pow,nprev,nn,r,r)

         OPEN(UNIT=10,FILE="shifted_points.txt")

         do i = 1,(nn - nprev)
            READ(10,*) Y(i,:,r)
         end do
         CLOSE(10,STATUS='DELETE')

      end do

   else

      ALLOCATE(array_chunks(num_cores))    !allocating array to hold chunks in                                                            
      chunks = floor(dble(M_iter)/num_cores)  !tells us the amount of chunks for each core                                                          
      rrr = mod(M_iter,num_cores)                !getting the remainder                                                                   

      !filling the array_chunks with chunks                     
      array_chunks(1) = chunks

      do i=2,size(array_chunks)
         array_chunks(i) = chunks
      end do

      !distributing the remainder if one exists, in a load balanced way                                                                   
      qq = 1
      do while(rrr /= 0)
         array_chunks(qq) = array_chunks(qq) + 1
         qq = qq + 1
         rrr = rrr - 1
      end do

      if((M_iter) < num_cores)then
         array_chunks = 0
         rrr = M_iter
         qq = 1
         do while(rrr /= 0)
            array_chunks(qq) = array_chunks(qq) + 1
            qq = qq + 1
            rrr = rrr - 1
         end do
      end if

      i = 1
      do i = 1,size(array_chunks)
         if(array_chunks(i) <= 0)then
            exit
         end if
      end do

      num_cores_used = i-1

      ALLOCATE(Y(nn - nprev,s_dim,M_iter))

      do r = 1,M_iter
         !applying a digital shift to the sobol points               
         call digitalshift(s_dim,M_iter,N_pow,N_pow,nprev,nn,r,r)

         OPEN(UNIT=10,FILE="shifted_points.txt")

         do i = 1,(nn - nprev)
            READ(10,*) Y(i,:,r)
         end do
         CLOSE(10,STATUS='DELETE')

      end do

   end if
end if

if((my_rank /= 0) .AND. (num_cores > M_iter))then
   allocate(array_chunks(M_iter))
else if((my_rank /= 0) .AND. (num_cores <= M_iter))then
   allocate(array_chunks(num_cores))
end if

if(num_cores > M_iter)then
   CALL MPI_Bcast(array_chunks,M_iter,MPI_INTEGER,master,MPI_COMM_WORLD,ierror)

   if(my_rank /= 0)then
      allocate(groups(M_iter))
   end if

   CALL MPI_Bcast(groups,M_iter,MPI_INTEGER,master,MPI_COMM_WORLD,ierror)

   if((0 <= my_rank) .AND. (my_rank<= groups(1)))then
      color = 1
   end if

   do i =2,M_iter
      if((groups(i-1)+1 <= my_rank) .AND. (my_rank<= groups(i)))then
         color = i
      end if
   end do

   Call MPI_COMM_SPLIT(MPI_COMM_WORLD,color,0,my_comm,ierror)
   call MPI_COMM_RANK(my_comm,my_new_rank,ierror)

   if(my_rank == 0)then
      do i=1,M_iter-1
         call mpi_send(Y(:,:,i+1),(nn-nprev)*s_dim,MPI_DOUBLE_PRECISION,groups(i)+1,groups(i) + 1,MPI_COMM_WORLD,ierror)
      end do

      ALLOCATE(aa2(nn-nprev,s_dim,1))
      aa2(:,:,1) = Y(:,:,1)

   end if

   if((my_rank /= 0) .AND. (my_new_rank == 0))then
      ALLOCATE(aa2(nn-nprev,s_dim,1))
      call MPI_RECV(aa2,(nn-nprev)*s_dim,MPI_DOUBLE_PRECISION,0,my_rank,MPI_COMM_WORLD,status,ierror)
   end if

   deallocate(groups)

   CALL MPI_BARRIER(MPI_COMM_WORLD,ierror)

   chunks = floor(dble(nn-nprev)/dble(array_chunks(color)))
   rrr = mod(nn-nprev,array_chunks(color))
   allocate(arr_c2(array_chunks(color)))
   arr_c2 = chunks

   qq = 1
   do while(rrr /= 0)
      arr_c2(qq) = arr_c2(qq) + 1
      qq = qq + 1
      rrr = rrr - 1
   end do

   allocate(aa(arr_c2(my_new_rank+1),s_dim))

   if((nn-nprev) > 1)then

      if(my_new_rank == 0)then
         do i = 1,array_chunks(color) - 1
            call mpi_send(aa2(arr_c2(i)+1:arr_c2(i)+arr_c2(i+1),:,1),s_dim*arr_c2(i),MPI_DOUBLE_PRECISION,i,i,my_comm,ierror)
         end do

         aa = aa2(1:arr_c2(1),:,1)
      else
         call MPI_RECV(aa,arr_c2(my_new_rank)*s_dim,MPI_DOUBLE_PRECISION,0,my_new_rank,my_comm,status,ierror)
      end if

   else
      aa = aa2(1:arr_c2(1),:,1)

   end if

   CALL MPI_BARRIER(MPI_COMM_WORLD,ierror)

   if(my_new_rank == 0)then
      DEALLOCATE(aa2)
   end if

else

   CALL MPI_Bcast(array_chunks,num_cores,MPI_INTEGER,master,MPI_COMM_WORLD,ierror)

   if(my_rank == 0)then
      do i=1,num_cores_used - 1
         start = startfun(array_chunks,i)
         ende = endfun(array_chunks,i)
         call mpi_send(Y(:,:,start:ende),(nn-nprev)*s_dim*array_chunks(i + 1),MPI_DOUBLE_PRECISION,i,i*2,MPI_COMM_WORLD,ierror)
      end do

      start = startfun(array_chunks,0)
      ende = endfun(array_chunks,0)
      ALLOCATE(aa2(nn-nprev,s_dim,array_chunks(my_rank + 1)))
      aa2 = Y(:,:,start:ende)

   end if

   if(my_rank /= 0)then
      ALLOCATE(aa2(nn-nprev,s_dim,array_chunks(my_rank + 1)))
      call MPI_RECV(aa2,(nn-nprev)*s_dim*array_chunks(my_rank + 1),MPI_DOUBLE_PRECISION,0,my_rank*2,MPI_COMM_WORLD,status,ierror)
   end if

   CALL MPI_BARRIER(MPI_COMM_WORLD,ierror)

end if

if(my_rank == 0)then
   DEALLOCATE(Y)
end if

ALLOCATE(acc1(M_iter),acc2(M_iter))

acc1 = dble(0)
acc2 = dble(0)

!setting constants                                                   
pi = dble(4)*atan(dble(1))

!sums for each vector                                                
sum1 = 0

if(num_cores > M_iter)then

   ALLOCATE(Pf_vec(arr_c2(my_new_rank+1)))
   Pf_vec = 0
   N_end1 = grids(ell + 1)
   M_end1 = N_end1

   ALLOCATE(random_vars2(arr_c2(my_new_rank+1),s_dim))

   ALLOCATE(rs(s_dim))
   
   counter2 = 1

   !running through the standard monte carlo method                  
   do i = 1,arr_c2(my_new_rank+1)
      Pf = 0

      rs = aa(i,:)

      random_vars2(i,:) = rs

      call numerical_model(rs,Pf,N_end1,M_end1)

      Pf_vec(counter2) = Pf
      acc1(color) = acc1(color) + Pf
      counter2 = counter2 + 1
   end do

   DEALLOCATE(aa)

   ALLOCATE(Pc_vec(arr_c2(my_new_rank+1)))
   Pc_vec = 0

   counter2 = 1
   N_end1 = (N_end1-1)/2 + 1
   M_end1 = (M_end1-1)/2 + 1

   do i = 1,arr_c2(my_new_rank+1)
      if(ell == 0)then
         Pc = 0
      else
         !initializing the output quantity of interest               
         Pc = 0
         rs = random_vars2(i,:)

         call numerical_model(rs,Pc,N_end1,M_end1)

      end if

      Pc_vec(counter2) = Pc
      acc2(color) =acc2(color) + Pc
      counter2 = counter2 + 1
   end do

   DEALLOCATE(random_vars2)

   ALLOCATE(temp_P(arr_c2(my_new_rank+1)))

   temp_P = Pf_vec - Pc_vec

   DEALLOCATE(arr_c2,rs)

   DEALLOCATE(Pf_vec,Pc_vec)

   !obtaining the sum for this problem to be used in mlmc            
   sum1(1) =  sum(temp_P)
   sum1(2) =  sum(temp_P**2)
   sum1(3) =  sum(temp_P**3)
   sum1(4) =  sum(temp_P**4)

   DEALLOCATE(temp_P)

else
   if(array_chunks(my_rank+1) > 0)then
      color = 1
   else
      color = 2
   end if

   if(color == 1)then

      ALLOCATE(Pf_vec((nn-nprev)*array_chunks(my_rank+1)))
      Pf_vec = 0
      N_end1 = grids(ell + 1)
      M_end1 = N_end1

      ALLOCATE(random_vars(array_chunks(my_rank+1),(nn-nprev),s_dim))

      ALLOCATE(rs(s_dim))

      counter = 1
      counter2 = 1
      start = startfun(array_chunks,my_rank)
      ende = endfun(array_chunks,my_rank)

      do r = start,ende

         !running through the standard monte carlo method            
         do i = 1,(nn-nprev)
            Pf = 0

            rs = aa2(i,:,counter)

            random_vars(counter,i,:) = rs

            call numerical_model(rs,Pf,N_end1,M_end1)

            Pf_vec(counter2) = Pf
            acc1(r) = acc1(r) + Pf
            counter2 = counter2 + 1
         end do

         counter = counter + 1

      end do

      DEALLOCATE(aa2)
 
      ALLOCATE(Pc_vec((nn-nprev)*array_chunks(my_rank+1)))
      Pc_vec = 0

      counter = 1
      counter2 = 1
      N_end1 = (N_end1-1)/2 + 1
      M_end1 = (M_end1-1)/2 + 1

      do r = start,ende

         do i = 1,(nn-nprev)
            if(ell == 0)then
               Pc = 0
            else
               !initializing the output quantity of interest         
               Pc = 0
               rs = random_vars(counter,i,:)

               call numerical_model(rs,Pc,N_end1,M_end1)

            end if

            Pc_vec(counter2) = Pc
            acc2(r) =acc2(r) + Pc
            counter2 = counter2 + 1
         end do

         counter = counter + 1

      end do

      DEALLOCATE(rs,random_vars)

   end if

   ALLOCATE(temp_P((nn-nprev)*array_chunks(my_rank+1)))

   temp_P = Pf_vec - Pc_vec

   DEALLOCATE(Pf_vec,Pc_vec)

   !obtaining the sum for this problem to be used in mlmc            
   sum1(1) =  sum(temp_P)
   sum1(2) =  sum(temp_P**2)
   sum1(3) =  sum(temp_P**3)
   sum1(4) =  sum(temp_P**4)

   DEALLOCATE(temp_P)

end if

DEALLOCATE(array_chunks)

if(num_cores > m_iter)then
   call MPI_COMM_FREE(my_comm,ierror)
end if

if(my_rank == 0)then
   OPEN(UNIT=10,FILE="shifts.txt")
   CLOSE(10,STATUS='DELETE')
end if

ALLOCATE(temp_diff(M_iter))

temp_diff = acc1 - acc2

CALL MPI_BARRIER(MPI_COMM_WORLD,ierror)

DEALLOCATE(acc1,acc2)

ALLOCATE(final_acc(M_iter))

CALL MPI_BARRIER(MPI_COMM_WORLD,ierror)

call MPI_REDUCE(temp_diff,final_acc,M_iter,MPI_DOUBLE_PRECISION,MPI_SUM,master,MPI_COMM_WORLD,ierror)

DEALLOCATE(temp_diff)

call MPI_REDUCE(sum1,final_sum1,4,MPI_DOUBLE_PRECISION,MPI_SUM,master,MPI_COMM_WORLD,ierror)

if(my_rank == master)then
   diff_acc(ell+1,:) = final_acc + diff_acc(ell+1,:)
end if

DEALLOCATE(final_acc)

CALL MPI_Bcast(final_sum1,4,MPI_DOUBLE_PRECISION,master,MPI_COMM_WORLD,ierror)

CALL MPI_Bcast(diff_acc,size(diff_acc(:,1))*M_iter,MPI_DOUBLE_PRECISION,master,MPI_COMM_WORLD,ierror)

CALL MPI_BARRIER(MPI_COMM_WORLD,ierror)

end subroutine qsampler


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


end MODULE mlqmc_mod
