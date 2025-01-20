#include <iostream>
#include "moving.h"
#include "exact.h"
#include <vector>
#include <ctime>
using namespace std;

int main()
{

  double beta=-2,T=1.5, acon = 0.25;
  double k = 1.0;
  double nu = 1.0;
  double c = 1.0;
  double k_c = 1.0;
  double dt; 
  int num_xpts, num_tpts, pt_size=8; 
  clock_t begin2, end2;
  int pts_vec[pt_size] = {21,41,81,161,321,641,1281,2561};//,5121,10241}; //,20481};
  double relative_fro_vec[pt_size], relative_conv_vec[pt_size-1];
  double elapsed_secs2, fro_norm_diff, fro_norm_exact;

  for(int pts_index=0;pts_index<pt_size;pts_index++)
    {
      num_xpts = pts_vec[pts_index];
      num_tpts = pts_vec[pts_index];
      dt = T/(double)(num_tpts-1);

      begin2 = clock();
      moving *bran2 = new moving(num_xpts,beta,dt,acon,k,nu,c,k_c,num_tpts);
      end2 = clock();

      exact *bran = new exact(num_xpts,beta,dt,acon,k,nu,c,k_c,num_tpts);

      // Computing Frobenius Norms
      fro_norm_exact = double(0);
      fro_norm_diff = double(0);
      for(int j=0;j<num_tpts;j++)
	{
	  for(int i=0;i<num_xpts;i++)
	    {
	      fro_norm_exact += pow(abs(bran->Exact_solution[i][j]), 2);
	      fro_norm_diff += pow(abs(bran->Exact_solution[i][j] - bran2->U_solution[i][j]), 2);
	    }
	}

      fro_norm_exact = sqrt(fro_norm_exact);
      fro_norm_diff = sqrt(fro_norm_diff);

      cout << "fro_norm " << fro_norm_diff/fro_norm_exact << endl; 

      bran->destroy_Exact_total(num_xpts); 
      delete bran;

      bran2->destroy_U_total(num_xpts);
      delete bran2;
  
      relative_fro_vec[pts_index] = fro_norm_diff/fro_norm_exact;

      elapsed_secs2 = double(end2 - begin2) / CLOCKS_PER_SEC;
      cout << "Elapsed time = " << elapsed_secs2 << endl;

    }

  cout << endl;
  cout << "relative_fro_vec" << endl; 
  for(int i=0;i<pt_size;i++)
    {
      cout << relative_fro_vec[i] << endl; 
    }
  cout << endl;

  for(int i=1;i<pt_size;i++)
    {
      relative_conv_vec[i-1] = log(relative_fro_vec[i-1]/relative_fro_vec[i])/log((double)(2));
    }

  cout << "relative_conv_vec" << endl;
  for(int i=0;i<pt_size-1;i++)
    {
      cout << relative_conv_vec[i] << endl; 
    }
  cout << endl; 

  return 0; 
}

