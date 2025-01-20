#include "exact.h"
#include <cmath>
#include <ctime>
#include <complex> 
using namespace std;

//default constructor 
exact::exact()
{  
}

//parameterized constructor
exact::exact(int num_xpts, double beta, double dt, double acon, double k, double nu, double c, double k_c, int num_tpts) 
{
  //variables to make matrix creation easier
  complex<double> eye(0.0,1.0);
  complex<double> c_zero(0.0,0.0); 
  double alpha = nu/c;
  double w = (pow(k, 2))*alpha;
  complex<double> constant((alpha*dt)/(double)2,0.0);

  //Creating a container to hold the full solution 
  Exact_solution = new complex<double>*[num_xpts];
  for(int i=0;i<num_xpts;i++)
  {
    Exact_solution[i] = new complex<double>[num_tpts];
  }
  
  //filling the entries for the initial solution 
  //complex<double> *U_vec = new complex<double>[num_xpts-2];

  for(int j=0;j<num_tpts;j++)
    {
      //Create the grid for the jth step
      create_grid(num_xpts, beta, dt, j, acon);

      for(int i=0;i<num_xpts-2;i++)
	{
	  //Filling in exact solution 
	  //*(U_vec+i) = exp(eye*k*(*(a+i+1)))*sin(k_c*M_PI*(*(a+i+1))*(*(a+i+1) - beta));
	  Exact_solution[i+1][j] = exp(eye*(k*(*(xm_grid+i+1)) - w*(j*dt)))*sin(k_c*M_PI*(*(xm_grid+i+1) - gamma(j*dt, acon))*(*(xm_grid+i+1) - beta));
	  Exact_solution[0][j] = c_zero;
	  Exact_solution[num_xpts-1][j] = c_zero;
	}

      //freeing the grid points for the jth step 
      destroy_grid();

    }

}

//Freeing the memory for the total solution
void exact::destroy_Exact_total(int num_xpts)
{
  for(int i=0;i<num_xpts;i++)
    {
      delete [] Exact_solution[i];
    }
  delete [] Exact_solution;
}

//creates the grids for evaluation of exact solution
void exact::create_grid(int num_xpts, double beta, double dt, int t_step, double acon)
{
  //m spatial step size
  double dxm = (gamma(t_step*dt,acon)-beta)/(double)(num_xpts-1);

  //grid points at time m
  xm_grid = new double[num_xpts];

  //creating the grid points
  for(int i=0;i<num_xpts;i++)
    {
      *(xm_grid+i) = beta + i*dxm;
    }

}

//freeing the memory for the created grid points
void exact::destroy_grid()
{
  delete [] xm_grid;
}

//function that determines the shutter position                 
double exact::gamma(double t, double acon)
{
  return acon*pow(t,2);
}





