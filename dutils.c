#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#define MAX_LENGTH_LINE  256

#define Aref(a1,a2)  A[ (a1-1)*(Alda)+(a2-1) ]
#define Bref(a1,a2)  B[ (a1-1)*(Blda)+(a2-1) ]
#define xref(a1)     x[ (a1-1) ]
#define yref(a1)     y[ (a1-1) ]
#define dref(a1)     d[ (a1-1) ]

/*===========================================================================*/
double dclock() {
/* 
 * Timer
 *
 */
  struct timeval  tv;
  struct timezone tz;

  gettimeofday( &tv, &tz );   

  return (tv.tv_sec + tv.tv_usec*1.0e-6);
}
/*===========================================================================*/
int read_data( char *filename, int *m, int *n, int *visual, double *tmin,
               int *random, double *init, double *incr ) {
/* 
 * Read problem data
 *
 *
 * m      : First dimension 
 * n      : Second dimension
 * visual : Print data/results
 * tmin   : Timer threshold
 * random : Generate random values?
 * init   : Initial value
 * incr   : Increment of the values
 *
 */
  FILE   *fp;
  char   myLine[ MAX_LENGTH_LINE ];

  if ((fp = fopen( filename, "r" ) ) == NULL) {
    printf( "ERROR: file %s does not exist.\n", filename );
    return -1;
  }


  /* Read problem dimensions */
  fscanf( fp, "%d", m );
  fgets( myLine, MAX_LENGTH_LINE, fp );
  if ( *m <= 0 ) {
    printf( "ERROR: m must be >= 1.\n" );
    return (-1);
  }
  fscanf( fp, "%d", n );
  fgets( myLine, MAX_LENGTH_LINE, fp );
  if ( *n <= 0 ) {
    printf( "ERROR: n must be >= 1.\n" );
    return (-1);
  }

  /* Read visualization option */
  fscanf( fp, "%d", visual );
  fgets( myLine, MAX_LENGTH_LINE, fp );
  if (( *visual != 0 )&&( *visual != 1 )) {
    printf( "ERROR: visual must be 0 or 1.\n" );
    return 1;
  }

  /* Read timing threshold */
  fscanf( fp, "%lg", tmin );
  fgets( myLine, MAX_LENGTH_LINE, fp );
  if ( *tmin < 0.0 ) {
    printf( "ERROR: timing threshold must be >= 0.0.\n" );
    return 1;
  }
  
  /* Read random indicator */
  fscanf( fp, "%d", random );
  fgets( myLine, MAX_LENGTH_LINE, fp );
  if (( *random != 0 )&&( *random != 1 )) {
      printf( "ERROR: random must be 0 or 1.\n" );
    return 1;
  }

  /* Read initial value of the generator */
  fscanf( fp, "%lg", init );
  fgets( myLine, MAX_LENGTH_LINE, fp );
  
  /* Read increment of the values for the generator */
  fscanf( fp, "%lg", incr );
  fgets( myLine, MAX_LENGTH_LINE, fp );

  return 0;

}
/*===========================================================================*/
int copy_matrix( int m, int n, double *A, int Alda, double *B, int Blda  ) {
  /*
   * Copy a matrix from the entries in other matrix
   * m      : Row dimension
   * n      : Column dimension
   * A      : Source Matrix
   * Alda   : Leading dimension of A
   * B      : Destination Matrix
   * Blda   : Leading dimension of B
   *
   */
  int i, j;
  
  for ( i=1; i<=m; i++ ) {
    for ( j=1; j<=n; j++ ) {
      Bref(i,j) = Aref(i,j);
    }
  }
  
  return 0;
}
/*===========================================================================*/
int generate_matrix( int m, int n, double *A, int Alda, double init, double incr ) {
/*
 * Generate a matrix with determinated entries
 * m      : Row dimension
 * n      : Column dimension
 * A      : Matrix
 * Alda   : Leading dimension
 * init   : First value
 * incr   : Increment of the values
 *
 */
  int i, j, k;
  double val = init;

  for ( i=1; i<=m; i++ ) {
    k = i;
    for ( j=1; j<=n; j++ ) {
      Aref(i,k) = val; val += incr;
      k = ( k == n )? 1: (k + 1);
    }
  }

  return 0;
}
/*===========================================================================*/
int generate_matrix_random( int m, int n, double *A, int Alda ) {
  /*
   * Generate a matrix with random entries
   * m      : Row dimension
   * n      : Column dimension
   * A      : Matrix
   * Alda   : Leading dimension
   *
   */
  int i, j;
  
  for ( i=1; i<=m; i++ )
    for ( j=1; j<=n; j++ )
      Aref(i,j) = ((double) rand())/RAND_MAX;
  
  return 0;
}
/*===========================================================================*/
int print_matrix( char *name, int m, int n, double *A, int Alda ) {
/*
 * Print a matrix to standard output
 * name   : Label for matrix name
 * m      : Row dimension
 * n      : Column dimension
 * A      : Matrix
 * Alda   : Leading dimension
 *
 */
  int i, j;

  for ( i=1; i<=m; i++ )
    for ( j=1; j<=n; j++ )
      printf( "   %s(%d,%d) = %22.15e;\n", name, i, j, Aref(i,j) );

  return 0;
}
/*===========================================================================*/
int copy_vector( int m, double *x, double *y  ) {
  /*
   * Copy a vector from the entries in other vector
   * m      : Dimension
   * x      : Source Vector
   * y      : Destination Vector
   *
   */
  int i;
  
  for ( i=1; i<=m; i++ ) {
    yref(i) = xref(i);
  }
  
  return 0;
}
/*===========================================================================*/
int generate_vector( int m, double *x, double init, double incr  ) {
  /*
   * Generate a vector with determinated entries
   * m      : Dimension
   * x      : Vector
   * init   : First value
   * incr   : Increment of the values
   *
   */
  int i;
  double val = init;
  
  for ( i=1; i<=m; i++ ) {
    xref(i) = val; val += incr;
  }
  
  return 0;
}
/*===========================================================================*/
int generate_vector_random( int m, double *x ) {
/*
 * Generate a vector with random entries
 * m      : Dimension
 * x      : Vector
 *
 */
  int i;

  for ( i=1; i<=m; i++ )
    xref(i) = ((double) rand())/RAND_MAX;

  return 0;
}
/*===========================================================================*/
int print_vector( char *name, int m, double *x ) {
/*
 * Imprimir un vector por la salida estándar
 * name   : Label for vector name
 * m      : Dimension
 * x      : Vector
 *
 */
  int i;

  for ( i=1; i<=m; i++ )
    printf( "   %s(%d) = %22.15e;\n", name, i, xref(i) );

  return 0;
}
/*===========================================================================*/
int generate_ivector( int m, int *d, int init, int incr  ) {
  /*
   * Generate a vector with determinated entries
   * m      : Dimension
   * d      : Vector
   * init   : First value
   * incr   : Increment of the values
   *
   */
  int i;
  int val = init;
  
  for ( i=1; i<=m; i++ ) {
    dref(i) = val; val += incr;
  }
  
  return 0;
}
/*===========================================================================*/
int print_ivector( char *name, int m, int *d ) {
  /*
   * Imprimir un vector por la salida estándar
   * name   : Label for vector name
   * m      : Dimension
   * d      : Vector
   *
   */
  int i;
  
  for ( i=1; i<=m; i++ )
    printf( "   %s(%d) = %8d;\n", name, i, dref(i) );
  
  return 0;
}
/*===========================================================================*/
int matrix_vector_product( int m, int n, double *A, int Alda, double *x , double *y ) {
  /*
   * Imprimir un vector por la salida estándar
   * name   : Label for vector name
   * m      : Dimension
   * x      : Vector
   *
   */
  int i, j;
  double val;
  
  for ( i=1; i<=m; i++ ) {
    val = 0.0;
    for ( j=1; j<=n; j++ ) {
      val += Aref(i,j) * xref(j);
    }
    yref(i) = val;
  }

  return 0;
}
/*===========================================================================*/
double compute_error( int m, double *x , double *y ) {
  /*
   * Imprimir un vector por la salida estándar
   * name   : Label for vector name
   * m      : Dimension
   * x      : Vector
   *
   */
  int i;
  double val = 0.0;
  
  for ( i=1; i<=m; i++ ) {
    val = 0.0;
    val += (xref(i)-yref(i)) * (xref(i)-yref(i));
  }
  
  return sqrt(val);
}
