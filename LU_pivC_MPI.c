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
	double* Af = NULL, * A = NULL, * xf = NULL, * x = NULL, * bf = NULL, * b = NULL, init, incr;
	double t1, t2, time, flops, tmin, piv, currVal;
	double timeLU = 0.0, timeTr = 0.0, GFLOPsLU, GFLOPsTr;
	int    i, j, k, nreps, info, m, n, visual, random, Alda, myId, size, bloque, n_bloque, receiveCount;
	int dim, iPiv, processWorking, processWorkingA, processWorkingB;
	int* vPiv = NULL, * sendFrom, * sizeToSend, * vtemp;
	MPI_Status st;
	MPI_Datatype col, colType;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myId);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (myId == ROOT){
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

	//Cálculo para reparto
	sendFrom = (int*)malloc(size * sizeof(int));
	sizeToSend = (int*)malloc(size * sizeof(int));

	bloque = n / size;
	n_bloque = n % size;
	
	//Calculo de como repartir las columnas de la matriz A y los elementos del vector B 
	for (i = 0; i < size; i++) {
		sendFrom[i] = (i == 0) ? 0 : sendFrom[i - 1] + sizeToSend[i - 1];
		sizeToSend[i] = (i < n_bloque) ? (bloque + 1) : bloque;
	}

	//Creacion de columna para reparto
	MPI_Type_vector(m, 1, n, MPI_DOUBLE, &col);
	MPI_Type_commit(&col);
	MPI_Type_create_resized(col, 0, sizeof(double), &colType);
	MPI_Type_commit(&colType);

	nreps = 0;
	time = 0.0;
	vtemp = (double*)malloc(n * sizeof(double));
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

		receiveCount = sizeToSend[myId];
		//Reparto de las columnas de la matrix correspondientes a cada proceso
		MPI_Scatterv(A, sizeToSend, sendFrom, colType, myId == ROOT ? MPI_IN_PLACE : &Aref(1, sendFrom[myId] + 1), receiveCount, colType, ROOT, MPI_COMM_WORLD);
		//Reparto de las columnas del vector correspondientes a cada proceso
		MPI_Scatterv(b, sizeToSend, sendFrom, MPI_DOUBLE, myId == ROOT ? MPI_IN_PLACE : &bref(sendFrom[myId] + 1), receiveCount, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
   
		processWorking = 0;
		for (k = 1; k <= dim; k++) {
			//Se determina a que proceso corresponde la columna que se esta tratando
			if (processWorking != size - 1 && k > sendFrom[processWorking + 1]) {
				processWorking++;
			}

			//El proceso al que corresponde la columna, calcula el máximo
			piv = dabs(Aref(k, k));
			iPiv = k;
			if (processWorking == myId) {
				currVal;
				for (i = k + 1; i <= m; i++) {
					currVal = dabs(Aref(i, k));
					if (piv < currVal) {
						piv = currVal;
						iPiv = i;
					}
				}
				dref(k) = iPiv;
				piv = Aref(iPiv, k);
			}			

			//Se distribuye el indice (fila) donde está el máximo
			MPI_Bcast(&iPiv, 1, MPI_INT, processWorking, MPI_COMM_WORLD);

			if (iPiv != k) {
				for (i = sendFrom[myId] + 1; i <= sendFrom[myId] + receiveCount; i++) {
					vtemp[i - 1] = Aref(k, i);
					Aref(k, i) = Aref(iPiv, i);
					Aref(iPiv, i) = vtemp[i - 1];
				}
        
				//Se determina que proceso tiene b(ipiv)
                int bPivLocation = 0;
                for (j = 1; j < size; j++) {
                	if (iPiv > sendFrom[j] && iPiv <= sendFrom[j] + sizeToSend[j]){
                    	bPivLocation = j;
                    	break;
					}
                }
         
		 		//Si el proceso que tiene la columna tambien tiene  b(ipiv), se hace el swap, si no se realiza el intercambio
                if (bPivLocation == processWorking){
                        int ptmp = bref(k);
                        bref(k) = bref(iPiv);
                        bref(iPiv) = ptmp;
                } else if (bPivLocation == myId){
                	MPI_Sendrecv_replace(&bref(iPiv), 1, MPI_DOUBLE, processWorking, TAG, processWorking, TAG, MPI_COMM_WORLD, &st);
                } else if (processWorking == myId) {
                	MPI_Sendrecv_replace(&bref(k), 1, MPI_DOUBLE, bPivLocation, TAG, bPivLocation, TAG, MPI_COMM_WORLD, &st);
                }
			}
      
			if (processWorking == myId) {
				for (i = k + 1; i <= m; i++) {
					Aref(i, k) /= piv;
				}
			}

			//Se distribuye la columna y el elemento del vector
			MPI_Bcast(&Aref(1, k), 1, colType, processWorking, MPI_COMM_WORLD);
            MPI_Bcast(&bref(k), 1, MPI_DOUBLE, processWorking, MPI_COMM_WORLD);       

			//Cada proceso realiza el calculo en sus columnas (limite inferior --> if, superior --> cond. segundo for)
			for (i = k + 1; i <= m; i++) {
				for (j = k + 1; j <= sendFrom[myId] + receiveCount; j++) {
                    if (j > sendFrom[myId]){
					    Aref(i, j) -= Aref(i, k) * Aref(k, j);
                    }
				}
				//Actualiza b(i) si lo tiene
                if (i > sendFrom[myId] && i <= sendFrom[myId] + receiveCount){
                    bref(i) -= Aref(i, k) * bref(k);
                }
			}
		}
   
        /*for (i = 1; i <= m; i++){
        for (j = sendFrom[myId] + 1; j <= sendFrom[myId] + receiveCount; j++){
            printf("%f ", Aref(i,j));
        }
        printf("\n");
        }
        printf("\n");
            for (j = sendFrom[myId] + 1; j <= sendFrom[myId] + receiveCount; j++){
        printf("%f ", bref(j));
        }
        printf("\n\n");*/

		MPI_Barrier(MPI_COMM_WORLD);
		t2 = MPI_Wtime();

		timeLU += (t2 > t1 ? t2 - t1 : 0.0);

		MPI_Barrier(MPI_COMM_WORLD);
		t1 = MPI_Wtime();

		//Backward substitution
		processWorkingA = size - 1;
		for (k = dim; k > 0; k--) {
			//Se determina que proceso tiene el elemento k del vector
			if (processWorkingA != 0 && k == sendFrom[processWorkingA]) {
				processWorkingA--;
			}
			processWorkingB = processWorkingA;
			for (i = k + 1; i <= n; i++) {
				//Se determina que proceso tiene la columna i
				if (processWorkingB != size - 1 && i > sendFrom[processWorkingB + 1]) {
					processWorkingB++;
				}
				int diffProcessWorking = processWorkingA != processWorkingB;
				double bk = bref(k);
				
				//Si los procesos son diferentes, A envia a B b(k), y B devuelve a A el nuevo valor de b(k)
				//Si los procesos son el mismo, no hay comunicacion
				if (processWorkingA == myId && diffProcessWorking) {
					MPI_Send(&bk, 1, MPI_DOUBLE, processWorkingB, TAG, MPI_COMM_WORLD);
					MPI_Recv(&bref(k), 1, MPI_DOUBLE, processWorkingB, TAG, MPI_COMM_WORLD, &st);
				}
				else if (processWorkingB == myId) {
					if (diffProcessWorking) {
						MPI_Recv(&bk, 1, MPI_DOUBLE, processWorkingA, TAG, MPI_COMM_WORLD, &st);
					}
					bk -= bref(i) * Aref(k, i);
					if (diffProcessWorking) {
						MPI_Send(&bk, 1, MPI_DOUBLE, processWorkingA, TAG, MPI_COMM_WORLD);
					}
                    else{
                        bref(k) = bk;
                    }
				}
			}

			if (processWorkingA == myId) {		
                bref(k) /= Aref(k, k);
			}
		}
   
		//Obtain the solution
		for (i = 1; i <= dim; i++) {
			xref(i) = bref(i);
		}
   
        //Root recoge los resultados parciales para comprobar el correcto resultado
        MPI_Gatherv(myId == ROOT ? MPI_IN_PLACE : &xref(sendFrom[myId] + 1), sizeToSend[myId], MPI_DOUBLE, x, sizeToSend, sendFrom, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

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

    if (myId == ROOT) {
        /* Print results */
        if (visual == 1) {
            print_matrix("Af", m, n, A, Alda);
            print_vector("xf", n, x);
            print_vector("bf", m, b);
        }
    
        printf("-->Results\n");
        printf("   Residual     = %12.6e\n", compute_error(m, x, xf));
        printf("   Time LU      = %12.6e seg.\n", timeLU);
        flops = ((double)n) * n * (m - n / 3.0);
        GFLOPsLU = flops / (1.0e+9 * timeLU);
        printf("   GFLOPs LU    = %12.6e     \n", GFLOPsLU);
    
        printf("   Time Tr      = %12.6e seg.\n", timeTr);
        flops = ((double)n) * n;
        GFLOPsTr = flops / (1.0e+9 * timeTr);
        printf("   GFLOPs Tr    = %12.6e     \n", GFLOPsTr);
    
    
        /* Free data */
        free(Af); free(A);
        free(xf); free(x);
        free(bf); free(b);
        free(vPiv);
		free(sizeToSend); free(sendFrom);
		free(vtemp);
        printf("End of program...\n");
        printf("----------------------------------------------------------\n");
    }
  
    MPI_Finalize();
	return 0;
}
