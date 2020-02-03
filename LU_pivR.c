#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define Aref(a1,a2)  A[ (a1-1)*(Alda)+(a2-1) ]
#define xref(a1)     x[ (a1-1) ]
#define bref(a1)     b[ (a1-1) ]
#define dabs(a)      ( (a) > 0.0 ? (a) : -(a) )
#define dref(a1)     vPiv[ (a1-1) ]

extern double dclock( void );
extern int    read_data( char *, int *, int *, int *, double *, int *, double *, double * );
extern int    print_matrix( char *, int, int, double *, int );
extern int    print_vector( char *, int, double * );
extern int    print_ivector( char *, int, int * );
extern int    copy_matrix( int, int, double *, int, double *, int );
extern int    generate_matrix( int, int, double *, int, double, double);
extern int    generate_matrix_random( int, int, double *, int );
extern int    copy_vector( int, double *, double * );
extern int    generate_vector( int, double *, double, double );
extern int    generate_vector_random( int, double * );
extern int    generate_ivector( int, int *, int, int );
extern int    matrix_vector_product( int, int, double *, int, double * , double * );
extern double compute_error( int m, double *x , double *y );

int main(int argc, char *argv[])
{
  double *Af = NULL, *A = NULL, *xf = NULL, *x = NULL, *bf = NULL, *b = NULL, init, incr;
  double t1, t2, time, flops, tmin;
  double timeLU = 0.0, timeTr = 0.0, GFLOPsLU, GFLOPsTr;
  int    i, j, k, nreps, info, m, n, visual, random, Alda;
  int    *vPiv = NULL;
  
  printf("----------------------------------------------------------\n");
  printf("Driver for the evaluation of LU factorization routines\n");
  printf("SIE032. Paralelismo, Clusters y Computación GRID\n");
  printf("Universidad Jaime I de Castellón\n");
  printf("October, 2019\n");
  printf("----------------------------------------------------------\n");
  printf("Program starts...\n");
  
  printf("-->Read data\n");
  info = read_data( "main.in", &m, &n, &visual, &tmin, &random, &init, &incr );
  if ( info != 0 ) {
    printf("Error in data read\n");
    exit( -1 );
  }
  
  /* Allocate space for data */
  Af   = (double *) malloc(m*n*sizeof(double));   Alda = m;
  A    = (double *) malloc(m*n*sizeof(double));
  xf   = (double *) malloc(n*sizeof(double));
  x    = (double *) malloc(n*sizeof(double));
  bf   = (double *) malloc(m*sizeof(double));
  b    = (double *) malloc(m*sizeof(double));
  vPiv = (int    *) malloc(m*sizeof(int   ));

  /* Generate random data */
  if (random) {
    generate_matrix_random( m, n, Af, Alda );
    generate_vector_random( n, xf );
    generate_vector_random( m, bf );
  } else {
    generate_matrix( m, n, Af, Alda, init, incr );
    generate_vector( n, xf, init, incr );
    generate_vector( m, bf, 0.0, 0.0 );
  }
  matrix_vector_product( m, n, Af, Alda, xf, bf );

  /* Print data */
  printf("   Problem dimension, m = %d, n = %d\n", m, n);
  if ( visual == 1 ){
    print_matrix ( "Ai", m, n, Af, Alda );
    print_vector ( "xi", n, x );
    print_vector ( "bi", m, bf );
  }
  
  printf("-->Solve problem\n");
  
  nreps = 0;
  time  = 0.0;
  double *vtemp = (double *) malloc (m * sizeof(double));
  while ( ( info == 0 ) && ( time < tmin )) {
    copy_matrix( m, n, Af, Alda, A, Alda );
    generate_vector( n, x, 0.0, 0.0 );
    copy_vector( m, bf, b );
    generate_ivector( m, vPiv, 1, 1 );

    /* LU factorization */
    t1   = dclock();

    int dim = m < n ? m : n;
    for (k = 1; k <= dim; k++) {
      double piv = dabs(Aref(k, k));
      int iPiv = k;
      double currVal;
      for (i = k + 1; i <= n; i++) {
        currVal = dabs(Aref(k, i));
        if (piv < currVal) {
          piv = currVal;
          iPiv = i;
        }
      }
      piv = Aref(k,iPiv);
      printf("Iteracion %d, ipiv --> %d\n", k, iPiv);

      if (iPiv != k) {
        for (i = 1; i <= m; i++) {
          vtemp[i-1] = Aref(i, k);
          Aref(i, k) = Aref(i, iPiv);
          Aref(i, iPiv) = vtemp[i-1];
        }
        int ptmp = dref(k);
        dref(k) = dref(iPiv);
        dref(iPiv) = ptmp;
      }

      for (i = k; i <= n; i++) {
        Aref(k, i) /= piv;
      }

      bref(k) /= piv;

      for (i = k + 1; i <= m; i++) {
        for (j = k + 1; j <= n; j++) {
          Aref(i, j) -= Aref(i, k) * Aref(k, j);
        }
        bref(i) -= Aref(i, k) * bref(k);
      }      
    }

    //print_vector( "b", m, b );

    t2   = dclock();

    timeLU += ( t2 > t1 ? t2 - t1 : 0.0 );

    t1   = dclock();
  
    // Backward substitution
    for (k = dim; k > 0; k--) {
      for (i = 1; i <= k - 1; i++) {
        bref(i) -= bref(k) * Aref(i, k); 
      }
    }
     
    // Remove permutation
    for (i = 1; i <= n; i++) {
      xref(dref(i)) = bref(i);
    }
    
    t2   = dclock();

    timeTr += ( t2 > t1 ? t2 - t1 : 0.0 );

    time = timeLU + timeTr;
    nreps++;
  }

  printf("-->Salgo while, %d\n", nreps);

  if ( info != 0 ) {
    printf("Error in problem solution\n");
    exit( -1 );
  }
  
  timeLU /= nreps;
  timeTr /= nreps;
  
  /* Print results */
  if (visual == 1) {
    print_matrix( "Af", m, n, A, Alda );
    print_vector( "xf", n, x );
    print_vector( "bf", m, b );
  }
  
  printf("-->Results\n");
  printf("   Residual     = %12.6e\n", compute_error ( m, x, xf )  );
  printf("   Time LU      = %12.6e seg.\n", timeLU  );
  flops   = ((double) n) * n * (m - n/3.0);
  GFLOPsLU  = flops / (1.0e+9 * timeLU );
  printf("   GFLOPs LU    = %12.6e     \n", GFLOPsLU  );
  
  printf("   Time Tr      = %12.6e seg.\n", timeTr  );
  flops   = ((double) n) * n;
  GFLOPsTr  = flops / (1.0e+9 * timeTr );
  printf("   GFLOPs Tr    = %12.6e     \n", GFLOPsTr  );

  
  /* Free data */
  free(Af  ); free(A   );
  free(xf  ); free(x   );
  free(bf  ); free(b   );
  free(vPiv);
  printf("End of program...\n");
  printf("----------------------------------------------------------\n");
  
  return 0;
}
