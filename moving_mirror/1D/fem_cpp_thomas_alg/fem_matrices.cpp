#include "fem_matrices.h"

//default constructor 
fem_matrices::fem_matrices()
{
}

//uses linear hat basis functions
//M_ij = integral_domain phi(x)*phi(x) dx
void fem_matrices::M_matrix_sub(double *M3, double *x, int n)
{
  //subdiagonal of M                          
  for(int j=1;j<n-2;j++)
    {
      *(M3+j-1) = (*(x+j+1)-*(x+j))/(double)(6);
    }
}

void fem_matrices::M_matrix_super(double *M2, double *x, int n)
{
  //superdiagonal of M                             
  for(int j=2;j<n-1;j++)
    {
      *(M2+j-2) = (*(x+j)-*(x+j-1))/(double)(6);
    }
}

void fem_matrices::M_matrix_diag(double *M1, double *x, int n)
{
  //diagonal of M                                
  for(int j=1;j<n-1;j++)
    {
      *(M1+j-1) = (*(x+j+1) - *(x+j-1))/(double)(3);
    }
}

//uses linear hat basis functions
//A_ij = integral_domain phi^\prime(x)*phi^\prime(x) dx
void fem_matrices::A_matrix_sub(double *A3, double *x, int n)
{
  //subdiagonal of A                                     
  for(int j=1;j<n-2;j++)
    {
      *(A3+j-1) = (double)(-1)/(*(x+j+1)-*(x+j));
    }
}

void fem_matrices::A_matrix_super(double *A2, double *x, int n)
{  
  //superdiagonal of A                                    
  for(int j=2;j<n-1;j++)
    {
      *(A2+j-2) = (double)(-1)/(*(x+j)- *(x+j-1));
    }
}
void fem_matrices::A_matrix_diag(double *A1, double *x, int n) 
{ 
  //diagonal of A                                
  for(int j=1;j<n-1;j++)
    {
      *(A1+j-1) = (double)(1)/(*(x+j) - *(x+j-1)) + (double)(1)/(*(x+j+1) - *(x+j));
    }  
}

//uses linear hat basis functions                            
//Q_ij = integral_domain phi^\prime(x)*(sum_r (xm - xm1)*phi_r(x))*phi(x) dx
void fem_matrices::Q_matrix_sub(double *Q3, double *xm1, double *xm, int n)
{
  //subdiagonal of Q                                     
  for(int j=2;j<n-1;j++)
    {
      *(Q3+j-2) = (-2*(*(xm+j)) + 2*(*(xm1+j))- *(xm+j-1) + *(xm1+j-1))/(double)(6);
    }
}

void fem_matrices::Q_matrix_super(double *Q2, double *xm1, double *xm, int n)
{
  //superdiagonal of Q                                 
  for(int j=1;j<n-1;j++) 
    {
      *(Q2+j-1) = (2*(*(xm+j)) - 2*(*(xm1+j)) + *(xm+j+1) - *(xm1+j+1))/(double)(6);
    }
}

void fem_matrices::Q_matrix_diag(double *Q1, double *xm1, double *xm, int n)
{
  //diagonal of Q                                    
  for(int j=1;j<n-1;j++)
    {
      *(Q1+j-1) = (*(xm+j-1) - *(xm1+j-1) - *(xm+j+1) + *(xm1+j+1))/(double)(6);
    }
}
