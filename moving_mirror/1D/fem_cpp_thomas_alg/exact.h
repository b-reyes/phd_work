//Class to creat exact solution of moving mirror problem
#ifndef exact_H
#define exact_H
#include <complex>
using namespace std;

class exact
{
 public:
  exact(); 
  exact(int num_xpts, double beta, double dt, double acon, double k, double nu, double c, double k_c, int num_tpts);
  complex<double> **Exact_solution;
  void destroy_Exact_total(int num_xpts);
  private: 
  double *xm_grid;
  double gamma(double t, double acon);
  void destroy_grid();
  void create_grid(int xgrid_spaces, double beta, double dt, int t_step, double acon);
};

#endif
