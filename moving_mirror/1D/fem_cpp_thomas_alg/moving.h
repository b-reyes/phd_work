//Class to conduct moving mirror problem
#ifndef MOVING_H
#define MOVING_H
#include "fem_matrices.h"
#include <complex>
using namespace std;

class moving : public fem_matrices
{
 public:
  moving(); 
  moving(int num_xpts, double beta, double dt, double acon, double k, double nu, double c, double k_c, int num_tpts);
  complex<double> **U_solution;
  void destroy_U_total(int num_xpts);
  private: 
  double *xm1_grid;
  double *xh_grid;
  double *xm_grid;
  double gamma(double t, double acon);
  void destroy_grid();
  void create_grid(int xgrid_spaces, double beta, double dt, int t_step, double acon);
  void create_sub(complex<double> *LHS3, complex<double> *RHS3, int num_xpts, complex<double> constant);
  void create_super(complex<double> *LHS2, complex<double> *RHS2, int num_xpts, complex<double> constant);
  void create_diag(complex<double> *LHS1, complex<double> *RHS1, int num_xpts, complex<double> constant);

  void lgwt(int N, double a, double b, double *x, double *ww);
  void g_vec(double *xh_grid, complex<double> *G, double sz_xx, double t, double w, double k, double beta, double acon, double alpha, double k_c);
  complex<double> evaluate_g(double x, double t, double w, double k, double beta, double acon, double alpha, double k_c);

  void thomas(complex<double> *sub, complex<double> *diag, complex<double> *super, complex<double> *soln, complex<double> *b, int n);
  void tri_diag_mat_vec_mult(complex<double> *RHS3, complex<double> *RHS1, complex<double> *RHS2, complex<double> *U_vec, complex<double> *Soln, complex<double> *G, double dt, int num_xpts);
};

#endif
