MODULE qmc_mod
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

subroutine qmc(N_pow,s_dim,shift_pow,N_end,M_end)
use iso_c_binding
implicit none
!declaring variables                                          
integer :: ierror,my_rank,num_cores,chunks,master = 0,i
double precision :: time1, time2
integer, intent(in) :: N_pow,s_dim,shift_pow,N_end,M_end
integer, allocatable,dimension(:) :: array_chunks,groups,arr_c2
double precision, allocatable, dimension(:,:) :: aa
integer :: num_cores_used,color,my_new_rank
double precision :: P_temp,P
integer :: rrr,qq,status(MPI_STATUS_SIZE)
integer :: end,start,my_comm,counter,counter2
integer :: N_qmc
integer(c_int) :: r,s,M_iter,nprev,nn
double precision, allocatable,dimension(:,:,:) :: Y,aa2
double precision, allocatable,dimension(:) :: acc,final_acc,temp1
double precision :: Q,varQ,acc_tot
character(len=21) :: names

CALL MPI_Comm_rank(MPI_COMM_WORLD,my_rank,ierror)
CALL MPI_COMM_Size(MPI_COMM_WORLD,num_cores,ierror)

CALL MPI_BARRIER(MPI_COMM_WORLD,ierror)

N_qmc = int(2)**N_pow

if(my_rank == master)then
   OPEN(unit=282,file='point_gen.txt')
   READ(282,*) names
   close(282) 
   print *, 'Deterministic point generator matrix file: ', names 
   print *, 'Number of samples:',N_qmc
   print *, 'Number of shifts: ',int(2)**shift_pow 
   print *, 'Stochastic dimension:',s_dim
   print *, 'N_end: ',N_end
   print *, 'M_end: ',M_end
   print *, 'Number of cores:',num_cores
   time1 = MPI_Wtime()
end if

M_iter = int(2)**shift_pow

s = int(s_dim,c_int) !Dimension                                 

nprev = 0

nn = N_qmc/M_iter

if(my_rank == 0)then
   !creating the shifts and writing them to a file                   
   call shiftgen(s,M_iter)

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
         call digitalshift(s,M_iter,N_pow,N_pow,nprev,nn,r,r)

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
         call digitalshift(s,M_iter,N_pow,N_pow,nprev,nn,r,r)

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
         end = endfun(array_chunks,i)
         call mpi_send(Y(:,:,start:end),(nn-nprev)*s_dim*array_chunks(i + 1),MPI_DOUBLE_PRECISION,i,i*2,MPI_COMM_WORLD,ierror)
      end do

      start = startfun(array_chunks,0)
      end = endfun(array_chunks,0)
      ALLOCATE(aa2(nn-nprev,s_dim,array_chunks(my_rank + 1)))
      aa2 = Y(:,:,start:end)

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

!initializing the output quantity of interest                        
P = 0

nprev = 0

ALLOCATE(acc(M_iter))
acc = 0

if(num_cores > M_iter)then

   counter2 = 1

   !initialize all constant variables for numerical model     
   call numerical_model_init

   !running through the standard monte carlo method                  
   do i = 1,arr_c2(my_new_rank+1)

      allocate(temp1(size(aa(i,:))))

      temp1 = aa(i,:)

      !running the numerical model
      call numerical_model(temp1,P_temp,N_end,M_end)

      DEALLOCATE(temp1)

      acc(color) = acc(color) + P_temp
      counter2 = counter2 + 1
   end do

   !deallocating all variables 
   call numerical_model_fin

   DEALLOCATE(aa)

else
   if(array_chunks(my_rank+1) > 0)then
      color = 1
   else
      color = 2
   end if

   if(color == 1)then
      counter = 1
      counter2 = 1
      start = startfun(array_chunks,my_rank)
      end = endfun(array_chunks,my_rank)
      
      !initialize all constant variables for numerical model    
      call numerical_model_init

      do r = start,end

         !running through the standard monte carlo method            
         do i = 1,(nn-nprev) 

            allocate(temp1(size(aa2(i,:,counter))))

            temp1 = aa2(i,:,counter)

            !running the numerical model
            call numerical_model(temp1,P_temp,N_end,M_end)

            DEALLOCATE(temp1)
            
            acc(r) = acc(r) + P_temp
            counter2 = counter2 + 1
         end do

         counter = counter + 1
      end do

      !deallocating all variables 
      call numerical_model_fin

      DEALLOCATE(aa2)

   end if
end if

if(num_cores > m_iter)then
   call MPI_COMM_FREE(my_comm,ierror)
end if

if(my_rank == 0)then
   ALLOCATE(final_acc(M_iter))
end if

CALL MPI_BARRIER(MPI_COMM_WORLD,ierror)

call MPI_REDUCE(acc,final_acc,M_iter,MPI_DOUBLE_PRECISION,MPI_SUM,master,MPI_COMM_WORLD,ierror)

DEALLOCATE(acc,array_chunks)

if(my_rank == 0)then
   time2 = MPI_Wtime()

   OPEN(UNIT=10,FILE="shifts.txt")
   CLOSE(10,STATUS='DELETE')
   acc_tot = 0 !calculate average over all shifts:                   

   do i = 1,M_iter
      acc_tot = acc_tot + final_acc(i)
   end do

   Q = acc_tot/dble(N_qmc)
   varQ = 0 !calculate variance of estimator Q(g)                    

   do i = 1,M_iter
      varQ = varQ + ((final_acc(i)/dble(nn)) - Q)**2
   end do

   DEALLOCATE(final_acc)

   varQ = varQ/dble(M_iter-1)
   varQ = sqrt(varQ)
   varQ = varQ/sqrt(dble(M_iter))

   print *, 'E[Q]: ',Q
   print *, 'S_E[Q]: ',varQ
   print *, 'exact solution time seconds (mult. by num_cores): ', (time2 - time1)*dble(num_cores)
   print *, ''

end if

end subroutine qmc

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


end MODULE qmc_mod
