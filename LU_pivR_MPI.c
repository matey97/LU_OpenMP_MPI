#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define Aref(a1,a2)  A[ (a1-1)*(Alda)+(a2-1) ]
#define xref(a1)     x[ (a1-1) ]
#define bref(a1)     b[ (a1-1) ]
#define dabs(a)      ( (a) > 0.0 ? (a) : -(a) )
#define dref(a1)     vPiv[ (a1-1) ]
#define TAG 53
#define ROOT 0

extern double dclock(void);
extern int    read_data(char*, int*, int*, int*, double*, int*, double*, double*);
extern int    print_matrix(char*, int, int, double*, int);
extern int    print_vector(char*, int, double*);
extern int    print_ivector(char*, int, int*);
extern int    copy_matrix(int, int, double*, int, double*, int);
extern int    generate_matrix(int, int, double*, int, double, double);
extern int    generate_matrix_random(int, int, double*, int);
extern int    copy_vector(int, double*, double*);
extern int    generate_vector(int, double*, double, double);
extern int    generate_vector_random(int, double*);
extern int    generate_ivector(int, int*, int, int);
extern int    matrix_vector_product(int, int, double*, int, double*, double*);
extern double compute_error(int m, double* x, double* y);

int main(int argc, char* argv[])
{
	double* Af = NULL, * A = NULL, * xf = NULL, * x = NULL, * bf = NULL, * b = NULL, init, incr, * vtemp;
	double t1, t2, time, flops, tmin, piv, currVal;
	double timeLU = 0.0, timeTr = 0.0, GFLOPsLU, GFLOPsTr;
	int    i, j, k, nreps, info, m, n, visual, random, Alda, myId, size, bloque, n_bloque, rowsToReceive;
	int dim, iPiv, processWorking, processWorkingA, processWorkingB;
	int* vPiv = NULL, * sendFromA, * sizeToSendA ;
	MPI_Status st;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myId);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (myId == ROOT) {
		printf("----------------------------------------------------------\n");
		printf("Driver for the evaluation of LU factorization routines\n");
		printf("SIE032. Paralelismo, Clusters y Computación GRID\n");
		printf("Universidad Jaime I de Castellón\n");
		printf("October, 2019\n");
		printf("----------------------------------------------------------\n");
		printf("Program starts...\n");
    	printf("-->Read data\n");
	}
 
  
	info = read_data("main.in", &m, &n, &visual, &tmin, &random, &init, &incr);
  	if (info != 0) {
		printf("Error in data read\n");
		exit(-1);
	}

	if (n > m) {
		if (myId == ROOT) {
			printf("This algorithm does not work for incompatible systems\n");
		}
		MPI_Finalize();
		exit(-1);
	}

	/* Allocate space for data */
	Af = (double*)malloc(m * n * sizeof(double));   Alda = m;
	A = (double*)malloc(m * n * sizeof(double));
	xf = (double*)malloc(n * sizeof(double));
	x = (double*)malloc(n * sizeof(double));
	bf = (double*)malloc(m * sizeof(double));
	b = (double*)malloc(m * sizeof(double));
	vPiv = (int*)malloc(m * sizeof(int));

	/* Generate random data */
	if (random) {
		generate_matrix_random(m, n, Af, Alda);
		generate_vector_random(n, xf);
		generate_vector_random(m, bf);
	}
	else {
		generate_matrix(m, n, Af, Alda, init, incr);
		generate_vector(n, xf, init, incr);
		generate_vector(m, bf, 0.0, 0.0);
	}
	matrix_vector_product(m, n, Af, Alda, xf, bf);

	/* Print data */
	if (myId == ROOT) {
		printf("   Problem dimension, m = %d, n = %d\n", m, n);
		if (visual == 1) {
			print_matrix("Ai", m, n, Af, Alda);
			print_vector("xi", n, x);
			print_vector("bi", m, bf);
		}

		printf("-->Solve problem\n");
	}

	//Cálculo para el reparto
	sendFromA = (int*)malloc(size * sizeof(int));
	sizeToSendA = (int*)malloc(size * sizeof(int));

	bloque = m / size;
	n_bloque = m % size;

	//Calculo de como repartir las filas de la matriz A y los elementos del vector B 
	for (i = 0; i < size; i++) {
		sendFromA[i] = (i == 0) ? 0 : sendFromA[i - 1] + sizeToSendA[i - 1];
		sizeToSendA[i] = (i < n_bloque) ? (bloque + 1) : bloque;
	}

	//Creacion de fila para reparto
	MPI_Datatype rowType;
	MPI_Type_contiguous(n, MPI_DOUBLE, &rowType);
	MPI_Type_commit(&rowType);

	nreps = 0;
	time = 0.0;
	vtemp = (double*)malloc(m * sizeof(double));
	while ((info == 0) && (time < tmin) && nreps != 1) {
		if (myId == ROOT){
			copy_matrix(m, n, Af, Alda, A, Alda);
			copy_vector(m, bf, b);
    	}
		generate_vector(n, x, 0.0, 0.0);
		generate_ivector(m, vPiv, 1, 1);

		/* LU factorization */
		MPI_Barrier(MPI_COMM_WORLD);
		t1 = MPI_Wtime();
		dim = m < n ? m : n;

		rowsToReceive = sizeToSendA[myId];
		//Reparto de las filas de la matriz correspondientes a cada proceso
		MPI_Scatterv(A, sizeToSendA, sendFromA, rowType, myId == ROOT ? MPI_IN_PLACE : &Aref(sendFromA[myId] + 1, 1), rowsToReceive, rowType, ROOT, MPI_COMM_WORLD);
		//Reparto de los elementos del vector correspondientes a cada proceso
		MPI_Scatterv(b, sizeToSendA, sendFromA, MPI_DOUBLE, myId == ROOT ? MPI_IN_PLACE : &bref(sendFromA[myId] + 1), sizeToSendA[myId], MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    
    	processWorking = 0;
		for (k = 1; k <= dim; k++) {
			//Se determina a que proceso corresponde la fila que se esta tratando
			if (processWorking != size - 1 && k > sendFromA[processWorking + 1]) {
				processWorking++;
			}
  
			//El proceso al que corresponde la fila, calcula el máximo
			piv = dabs(Aref(k, k));
			iPiv = k;
			if (processWorking == myId) {
				for (i = k + 1; i <= n; i++) {
					currVal = dabs(Aref(k, i));
					if (piv < currVal) {
						piv = currVal;
						iPiv = i;
					}
				}
				piv = Aref(k, iPiv);
			}

			//Se distribuye el indice (columna) donde está el máximo
			MPI_Bcast(&iPiv, 1, MPI_INT, processWorking, MPI_COMM_WORLD);

			//Todos los procesos realizan el swap en sus correspondientes filas
			if (iPiv != k) {
				for (i = sendFromA[myId] + 1; i <= sendFromA[myId] + rowsToReceive; i++) {
					vtemp[i - 1] = Aref(i, k);
					Aref(i, k) = Aref(i, iPiv);
					Aref(i, iPiv) = vtemp[i - 1];
				}
				int ptmp = dref(k);
				dref(k) = dref(iPiv);
				dref(iPiv) = ptmp;
			}


			if (processWorking == myId) {
				for (i = k; i <= n; i++) {
					Aref(k, i) /= piv;
				}
        		bref(k) /= piv;
			}
		
			//Se distribuye la fila y el elemento del vector
			MPI_Bcast(&Aref(k,1), n, MPI_DOUBLE, processWorking, MPI_COMM_WORLD);
      		MPI_Bcast(&bref(k), 1, MPI_DOUBLE, processWorking, MPI_COMM_WORLD);
  	
	  		//Cada proceso realiza el calculo en sus filas (limite inferior --> if, limite superior --> cond. primer for)
			for (i = k + 1; i <= sendFromA[myId] + rowsToReceive; i++) {
				if (i > sendFromA[myId]){
					for (j = k + 1; j <= n; j++) {
						Aref(i, j) -= Aref(i, k) * Aref(k, j);
					}
					bref(i) -= Aref(i, k) * bref(k);
				}
			}
		}
   
		/*printf("Proceso %d:\n", myId);
		for (i = sendFromA[myId]; i < sendFromA[myId] + rowsToReceive; i++) {
			for (j = 1; j <= n; j++) {
				printf("%f ", Aref(i + 1, j));
			}
			printf("\n");
		}
		printf("\n");
	
		for (i = sendFromB[myId]; i < sendFromB[myId] + sizeToSendB[myId]; i++){
		printf("%f ", bref(i + 1));
		}
		printf("\n");*/

		MPI_Barrier(MPI_COMM_WORLD);
		t2 = MPI_Wtime();

		timeLU += (t2 > t1 ? t2 - t1 : 0.0);

		MPI_Barrier(MPI_COMM_WORLD);
		t1 = MPI_Wtime();

		// Backward substitution
    	processWorkingA = size - 1;
		for (k = dim; k > 0; k--) {
			//Se determina que proceso tiene  el elemento k del vector
			if (processWorkingA != 0 && k == sendFromA[processWorkingA]) {
				processWorkingA--;
			}
      		processWorkingB = 0;
			for (i = 1; i <= k - 1; i++) {
				//Se determina que proceso tiene la fila i
				if (processWorkingB != size - 1 && i > sendFromA[processWorkingB + 1]) {
          			processWorkingB++;
        		}
				double bk = bref(k);
				//Si ambos procesos son distintos, A envia a B el elemento k del vector
				if (processWorkingA != processWorkingB) {
					if (processWorkingA == myId) {
						MPI_Send(&bk, 1, MPI_DOUBLE, processWorkingB, TAG, MPI_COMM_WORLD);
					}
					else if (processWorkingB == myId) {
						MPI_Recv(&bk, 1, MPI_DOUBLE, processWorkingA, TAG, MPI_COMM_WORLD, &st);
					}
				}
        
				//B realiza los calculos 
				if (processWorkingB == myId) {
					bref(i) -= bk * Aref(i, k);
				}
			}
		}
   
		//Root recoge los elementos del vector para realizar la permutación
    	MPI_Gatherv(myId == ROOT ? MPI_IN_PLACE : &bref(sendFromA[myId] + 1), sizeToSendA[myId], MPI_DOUBLE, b, sizeToSendA, sendFromA, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

		// Remove permutation
		for (i = 1; i <= n; i++) {
			xref(dref(i)) = bref(i);
		}
	
		//Reparto de las soluciones
    	MPI_Scatterv(x, sizeToSendA, sendFromA, MPI_DOUBLE, myId == ROOT ? MPI_IN_PLACE : &xref(sendFromA[myId] + 1), sizeToSendA[myId], MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
           
		MPI_Barrier(MPI_COMM_WORLD);
		t2 = MPI_Wtime();

		timeTr += (t2 > t1 ? t2 - t1 : 0.0);

		time = timeLU + timeTr;
		nreps++;
	}

	if (info != 0) {
		printf("Error in problem solution\n");
		exit(-1);
	}

	timeLU /= nreps;
	timeTr /= nreps;

	/* Print results */
	if (myId == ROOT) {
		if (visual == 1) {
			print_matrix("Af", m, n, A, Alda);
			print_vector("xf", n, x);
			print_vector("bf", m, b);
		}

		printf("-->Results\n");
        printf("   Residual     = %12.6e\n", i, compute_error(m, x, xf));
		printf("   Time LU      = %12.6e seg.\n", timeLU);
		flops = ((double)n) * n * (m - n / 3.0);
		GFLOPsLU = flops / (1.0e+9 * timeLU);
		printf("   GFLOPs LU    = %12.6e     \n", GFLOPsLU);

		printf("   Time Tr      = %12.6e seg.\n", timeTr);
		flops = ((double)n) * n;
		GFLOPsTr = flops / (1.0e+9 * timeTr);
		printf("   GFLOPs Tr    = %12.6e     \n", GFLOPsTr);
		printf("End of program...\n");
		printf("----------------------------------------------------------\n");
	}
	
	/* Free data */
	free(Af); free(A);
	free(xf); free(x);
	free(bf); free(b);
	free(vPiv);
	free(sendFromA); 
	free(sizeToSendA); 
	free(vtemp);
	
	MPI_Finalize();
	return 0;
}
