//Class to construct common fem matrices
#ifndef FEM_MATRICES_H
#define FEM_MATRICES_H
class fem_matrices
{
 public: 
  fem_matrices(); //default constructor
  void M_matrix_sub(double *M3, double *x, int n);
  void M_matrix_super(double *M2, double *x,int n);
  void M_matrix_diag(double *M1, double *x,int n);
  void A_matrix_sub(double *A3, double *x, int n);
  void A_matrix_super(double *A2, double *x,int n);
  void A_matrix_diag(double *A1, double *x,int n);
  void Q_matrix_sub(double *Q3, double *xm1, double *xm, int n);
  void Q_matrix_super(double *Q2, double *xm1, double *xm, int n);
  void Q_matrix_diag(double *Q1, double *xm1, double *xm, int n);
};

#endif
