#include "moving.h"
#include <cmath>
#include <ctime>
#include <limits>
#include <complex> 
#include <iostream>
using namespace std;
using namespace std::complex_literals;

//default constructor 
moving::moving()
{  
}

//parameterized constructor
moving::moving(int num_xpts, double beta, double dt, double acon, double k, double nu, double c, double k_c, int num_tpts) 
{
  //variables to make matrix creation easier
  complex<double> eye(0.0,1.0);
  complex<double> c_zero(0.0,0.0); 
  double alpha = nu/c;
  double w = (pow(k, 2))*alpha;
  complex<double> constant((alpha*dt)/(double)2,0.0);

  //creating the intial grid spacing 
  double dx = (-beta)/(double)(num_xpts-1);
  double *a = new double[num_xpts];

  for(int i=0;i<num_xpts;i++)
    {
      *(a+i) = beta + dx*i;
    }

  //Creating a container to hold the full solution 
  U_solution = new complex<double>*[num_xpts];
  for(int i=0;i<num_xpts;i++)
  {
    U_solution[i] = new complex<double>[num_tpts];
  }
  
  //filling the entries for the initial solution 
  complex<double> *U_vec = new complex<double>[num_xpts-2];

  for(int i=0;i<num_xpts-2;i++)
    {
      *(U_vec+i) = exp(eye*k*(*(a+i+1)))*sin(k_c*M_PI*(*(a+i+1))*(*(a+i+1) - beta));
      U_solution[i+1][0] = *(U_vec+i);
    }
  U_solution[0][0] = c_zero;
  U_solution[num_xpts-1][0] = c_zero; 

  //freeing the memory associated with the initial condition 
  delete [] a;

  //iterating through time steps 
  for(int m=1;m<num_tpts;m++)
    {
      //creates grid points at m-1,m-1.5, and m
      create_grid(num_xpts, beta, dt, m, acon); 

      complex<double> *G = new complex<double>[num_xpts-2];

      g_vec(&xh_grid[0], &G[0], num_xpts-2, (m*dt + (m-1)*dt )/(double)(2), w, k, beta, acon, alpha, k_c);

      complex<double> *LHS3 = new complex<double>[num_xpts-3];
      complex<double> *RHS3 = new complex<double>[num_xpts-3];

      create_sub(&LHS3[0],&RHS3[0],num_xpts,constant);

      complex<double> *LHS2 = new complex<double>[num_xpts-3];
      complex<double> *RHS2 = new complex<double>[num_xpts-3];

      create_super(&LHS2[0],&RHS2[0],num_xpts,constant);

      complex<double> *LHS1 = new complex<double>[num_xpts-2];
      complex<double> *RHS1 = new complex<double>[num_xpts-2];

      create_diag(&LHS1[0],&RHS1[0],num_xpts,constant);

      complex<double> *b = new complex<double>[num_xpts-2]; 

      //freeing the grid points 
      destroy_grid();

      //creating b in Ax=b
      tri_diag_mat_vec_mult(&RHS3[0],&RHS1[0],&RHS2[0],&U_vec[0],&b[0], &G[0], dt, num_xpts);

      delete [] RHS1;
      delete [] RHS2;
      delete [] RHS3;

      //solving Ax = b for the solution vector U_vec
      thomas(&LHS3[0],&LHS1[0],&LHS2[0],&U_vec[0],&b[0],num_xpts-2);
      delete [] LHS1;
      delete [] LHS2;
      delete [] LHS3;
      delete [] b; 
      delete [] G;
      
      //Putting the solution in 2D matrix for later manipulations 
      for(int i=1;i<num_xpts-1;i++)
      	{
	  U_solution[i][m] = *(U_vec+i-1);
      	}
      U_solution[0][m] = c_zero;
      U_solution[num_xpts-1][m] = c_zero;
      
    }

  delete [] U_vec;
}

//Freeing the memory for the total solution
void moving::destroy_U_total(int num_xpts)
{
  for(int i=0;i<num_xpts;i++)
    {
      delete [] U_solution[i];
    }
  delete [] U_solution;
}

//creates the grids for the mass,stiffness, and hybrid matrices
void moving::create_grid(int num_xpts, double beta, double dt, int t_step, double acon)
{
  //m-1 spatial step size                               
  double dxm1 = (gamma((t_step-1)*dt,acon)-beta)/(double)(num_xpts - 1);
  //m-1/2 spatial step size                              
  double dxh = (gamma(((t_step)*dt + (t_step-1)*dt )/(double)(2),acon)-beta)/(double)(num_xpts-1);
  //m spatial step size                                
  double dxm = (gamma(t_step*dt,acon)-beta)/(double)(num_xpts-1);

  //grid points at time m-1  
  xm1_grid = new double[num_xpts];
  //grid points at time m-1/2 
  xh_grid = new double[num_xpts];
  //grid points at time m                                
  xm_grid = new double[num_xpts];

  //creating the grid points                               
  for(int i=0;i<num_xpts;i++)
    {
      *(xm1_grid+i) = beta + i*dxm1;
      *(xh_grid+i) = beta + i*dxh;
      *(xm_grid+i) = beta + i*dxm;
    }
  
}

//freeing the memory for the created grid points 
void moving::destroy_grid()
{
  delete [] xm1_grid;
  delete [] xh_grid;
  delete [] xm_grid;
}

//function that determines the shutter position                 
double moving::gamma(double t, double acon)
{
  return acon*pow(t,2);
}

void moving::create_sub(complex<double> *LHS3, complex<double> *RHS3, int num_xpts, complex<double> constant)
{
  complex<double> eye(0.0,1.0);
  double *M3 = new double[num_xpts-3];
  double *A3 = new double[num_xpts-3];
  double *Q3 = new double[num_xpts-3];
  fem_matrices::M_matrix_sub(&M3[0],&xh_grid[0],num_xpts);
  fem_matrices::A_matrix_sub(&A3[0],&xh_grid[0],num_xpts);
  fem_matrices::Q_matrix_sub(&Q3[0],&xm1_grid[0],&xm_grid[0],num_xpts);

  for(int i=0;i<num_xpts-3;i++)
    {
      *(LHS3+i) = eye*(*(M3+i)) - (eye/(double)2)*(*(Q3+i)) - constant*(*(A3+i));
      *(RHS3+i) = eye*(*(M3+i)) + (eye/(double)2)*(*(Q3+i)) + constant*(*(A3+i));
    }
  
  delete [] M3;
  delete [] A3;
  delete [] Q3;
}

void moving::create_super(complex<double> *LHS2, complex<double> *RHS2, int num_xpts, complex<double> constant)
{
  complex<double> eye(0.0,1.0);
  double *M2 = new double[num_xpts-3];
  double *A2 = new double[num_xpts-3];
  double *Q2 = new double[num_xpts-3];
  fem_matrices::M_matrix_super(&M2[0],&xh_grid[0],num_xpts);
  fem_matrices::A_matrix_super(&A2[0],&xh_grid[0],num_xpts);
  fem_matrices::Q_matrix_super(&Q2[0],&xm1_grid[0],&xm_grid[0],num_xpts);

  for(int i=0;i<num_xpts-3;i++)
    {
      *(LHS2+i) = eye*(*(M2+i)) - (eye/(double)2)*(*(Q2+i)) - constant*(*(A2+i));
      *(RHS2+i) = eye*(*(M2+i)) + (eye/(double)2)*(*(Q2+i)) + constant*(*(A2+i));
    }
  
  delete [] M2;
  delete [] A2;
  delete [] Q2;
}

void moving::create_diag(complex<double> *LHS1, complex<double> *RHS1, int num_xpts, complex<double> constant)
{
  complex<double> eye(0.0,1.0);
  double *M1 = new double[num_xpts-2];
  double *A1 = new double[num_xpts-2];
  double *Q1 = new double[num_xpts-2];
  fem_matrices::M_matrix_diag(&M1[0],&xh_grid[0],num_xpts);
  fem_matrices::A_matrix_diag(&A1[0],&xh_grid[0],num_xpts);
  fem_matrices::Q_matrix_diag(&Q1[0],&xm1_grid[0],&xm_grid[0],num_xpts);

  for(int i=0;i<num_xpts-2;i++)
    {
      *(LHS1+i) = eye*(*(M1+i)) - (eye/(double)2)*(*(Q1+i)) - constant*(*(A1+i));
      *(RHS1+i) = eye*(*(M1+i)) + (eye/(double)2)*(*(Q1+i)) + constant*(*(A1+i)); 
    }
  
  delete [] M1;
  delete [] A1;
  delete [] Q1;
}

void moving::tri_diag_mat_vec_mult(complex<double> *RHS3, complex<double> *RHS1, complex<double> *RHS2, complex<double> *U_vec, complex<double> *Soln, complex<double> *G, double dt, int num_xpts)
{
  *(Soln) = (*(RHS1))*(*(U_vec)) + (*(RHS2))*(*(U_vec+1)) + dt*(*(G));

  for(int i=1;i<num_xpts-3;i++)
    {
      *(Soln+i) = (*(RHS3+i-1))*(*(U_vec+i-1)) + (*(RHS1+i))*(*(U_vec+i)) + (*(RHS2+i))*(*(U_vec+i+1)) + dt*(*(G + i));
    }

  *(Soln+num_xpts-3) = (*(RHS3+num_xpts-4))*(*(U_vec+num_xpts-4)) + (*(RHS1+num_xpts-3))*(*(U_vec+num_xpts-3)) + dt*(*(G +num_xpts-3));
}

//sub->lower diagonal of LHS                           
//diag->diagonal of LHS                               
//super->upper diagonal of LHS                             
//b->RHS of system of equations                            
//n->number of element in the diagonal of the LHS        
//soln->solution: LHS*soln = b                        
void moving::thomas(complex<double> *sub, complex<double> *diag, complex<double> *super, complex<double> *soln, complex<double> *b, int n)
{
  //Step 1: forward elimination                      
  for(int i=1;i<n;i++)
    {
      *(diag + i) = *(diag+i) - ( *(sub+i-1)/ *(diag+i-1) )*(*(super+i-1));
      *(b+i) = *(b+i) - ( *(sub+i-1)/ *(diag+i-1) )*(*(b+i-1));
    }

  //Step 2: back substitution                        
  *(b+n-1) = *(b+n-1) / *(diag+n-1);
  for(int j=n-2;j>-1;j--)
    {
      *(b+j) = ( *(b+j) - (*(super+j))*(*(b+j+1)) ) / *(diag+j);
    }

  //Filling solution vector                         
  for(int i=0;i<n;i++)
    {
      *(soln+i) = *(b+i);
    }
}

void moving::g_vec(double *xh_grid, complex<double> *G, double sz_xx, double t, double w, double k, double beta, double acon, double alpha, double k_c)
{

  complex<double> sum_x1, sum_x2;
  complex<double> g;

  for(int j=1; j<=sz_xx; j++)
    {

      double *x1 = new double[3]; 
      double *w1 = new double[3]; 
      lgwt(3, xh_grid[j - 1], xh_grid[j], &x1[0], &w1[0]);

      double *x2 = new double[3];
      double *w2 = new double[3];
      lgwt(3, xh_grid[j], xh_grid[j+1], &x2[0], &w2[0]);

      sum_x1 = 0i;
      for(int i=0; i<3; i++)
        {
	  sum_x1 += (((double)(1)/(xh_grid[j] - xh_grid[j-1]))*(evaluate_g(x1[i], t, w, k, beta, acon, alpha, k_c)*x1[i] - evaluate_g(x1[i], t, w, k, beta, acon, alpha, k_c)*xh_grid[j-1]))*w1[i];
	}

      sum_x2 = 0i;
      for(int i=0; i<3; i++)
        {
          sum_x2 += (((double)(1)/(xh_grid[j+1] - xh_grid[j]))*(evaluate_g(x2[i], t, w, k, beta, acon, alpha, k_c)*xh_grid[j+1] - evaluate_g(x2[i], t, w, k, beta, acon, alpha, k_c)*x2[i]))*w2[i];
	}

      G[j-1] = sum_x1 + sum_x2;

      delete [] x1; 
      delete [] w1; 
      delete [] x2;
      delete [] w2;

    }

}

complex<double> moving::evaluate_g(double x, double t, double w, double k, double beta, double acon, double alpha, double k_c)
{
  complex<double> g; 

  g = -exp(1i*(-t*w + k*x))*((double)(2)*1i*k_c*M_PI*(alpha*(1i + k*(beta + acon*(pow(t,2)) - (double)(2)*x)) + acon*t*(-beta + x))*cos(k_c*M_PI*(beta-x)*(-acon*(pow(t,2))+x)) + (-w+alpha*(pow(k,2) + (pow(k_c,2))*(pow(M_PI,2))*pow((beta + acon*(pow(t,2)) - (double)(2)*x),2)))*sin(k_c*M_PI*(-beta+x)*(-acon*(pow(t,2)) + x)));

  return g; 

}

void moving::lgwt(int N, double a, double b, double *x, double *ww)
{
  int N1, N2;
  N=N-1;
  N1=N+1;
  N2=N+2;

  double *xu = new double[N1]; 
  
  for(int i=0;i<N1;i++)
    {
      *(xu + i) = (double)(-1) + (((double)(1) - (double)(-1))/(double)(N1-1))*i;
    }


  // initial guess 
  double *y = new double[N1];
  double *y0 = new double[N1];
  double *y_diff = new double[N1];
  for(int i=0;i<N1;i++)
    {
      *(y + i) = cos((2*i+1)*M_PI/(2*N+2)) + (0.27/N1)*sin(M_PI*(*(xu + i))*N/N2);
      *(y0 + i) = (double)2;
    }

  // Legendre-Gauss Vandermonde Matrix
  double **L = new double*[N1];
  for(int i=0;i<N1;i++)
    {
      L[i] = new double[N2];
    }

  // Derivative of LGVM
  double *Lp = new double[N1];

  double max_y;
  max_y = abs(*(y) - *(y0));
  for(int i=1;i<N1;i++)
    {
      *(y_diff + i) = abs(*(y + i) - *(y0 + i)); 

      if(max_y < *(y_diff + i))
	{
	  max_y = *(y_diff + i);
	}
    }

  double eps = numeric_limits<double>::epsilon();

  // Iterate until new points are uniformly within epsilon of old points
  while(max_y > eps)
    {

      for(int i=0;i<N1;i++)
	{
	  L[i][0] = (double)1;
	  L[i][1] = *(y + i);
	  *(Lp + i) = (double)0;
	}

      for(int k=1;k<N1;k++)
	{
	  for(int i=0;i<N1;i++)
	    {
	      L[i][k+1] = ((double)(2*(k + 1) - 1)*(*(y+i))*L[i][k]- (double)(k)*L[i][k-1] )/(double)(k+1);
	    }
	}

      for(int i=0;i<N1;i++)
	{
	  *(Lp+i) = (double)(N2)*(L[i][N1-1]- (*(y+i))*L[i][N2-1] )/((double)1 - pow(*(y+i),2));  
	}
      
      for(int i=0;i<N1;i++)
	{
	  *(y0 + i) = *(y + i);
	}

      for(int i=0;i<N1;i++)
	{
	  *(y + i) = *(y0 + i) - L[i][N2-1]/(*(Lp + i));
	}

      max_y= abs(*(y) - *(y0));
      for(int i=1;i<N1;i++)
	{
	  *(y_diff + i) = abs(*(y + i) - *(y0 + i));

	  if(max_y < *(y_diff + i))
	    {
	      max_y = *(y_diff + i);
	    }
	}

    }

  // Linear map from[-1,1] to [a,b]
  for(int i=0;i<N1;i++)
    {
      *(x + i) = ( (double)(a)*((double)1 - *(y+i))+ (double)(b)*((double)1 + *(y+i)))/(double)2; 
    }

  // Compute the weights
  for(int i=0;i<N1;i++)
    {
      *(ww + i) = (double)(b-a)/(( (double)(1) - pow(*(y + i), 2))* pow(*(Lp + i), 2))*pow(((double)(N2)/(double)(N1)), 2);
    }

  delete [] xu;
  delete [] y;
  delete [] y0;
  delete [] y_diff;

  for(int i=0;i<N1;i++)
    {
      delete [] L[i];
    }

  delete [] L;
  delete [] Lp;
  
} 
