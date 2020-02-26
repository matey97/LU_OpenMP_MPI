#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

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
	double t1, t2, time, flops, tmin, piv;
	double timeLU = 0.0, timeTr = 0.0, GFLOPsLU, GFLOPsTr, partialResultProcess;
	int    i, j, k, nreps, info, m, n, visual, random, Alda, myId, size, bloque, n_bloque, receiveCount;
	int dim, iPiv, processWorking, processWorkingA, mpiGranted, partialResultsToReceive;
	int* vPiv = NULL, * sendFrom, * sizeToSend, startFrom;
	MPI_Status st;
	MPI_Datatype col, colType;

	MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpiGranted);
	MPI_Comm_rank(MPI_COMM_WORLD, &myId);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

    
    if (mpiGranted < MPI_THREAD_FUNNELED) {
        if (myId == ROOT) {
            printf("MPI_THREAD_FUNNELED not granted!\n");
        }
        MPI_Finalize();
        exit(-1);
    }

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
        printf("   Strategy: Column pivoting\n");
        printf("   Processes = %d, Threads per process = %s\n", size, getenv("OMP_NUM_THREADS"));
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
        startFrom = sendFrom[myId];

        //Recibo los datos en las estructuras de datos originales
		//Por ejemplo con 2 procesos, P1 recibira la segunda parte de la matriz en las columnas correspondientes de la estructura original
		//Se que no es eficiente a nivel de memoria, 
        //pero asi evito tener que lidiar con transformaciones en los indices y vivir unos añitos más :)

		//Reparto de las columnas de la matrix correspondientes a cada proceso
		MPI_Scatterv(A, sizeToSend, sendFrom, colType, myId == ROOT ? MPI_IN_PLACE : &Aref(1, startFrom + 1), receiveCount, colType, ROOT, MPI_COMM_WORLD);
		//Reparto de las columnas del vector correspondientes a cada proceso
		MPI_Scatterv(b, sizeToSend, sendFrom, MPI_DOUBLE, myId == ROOT ? MPI_IN_PLACE : &bref(startFrom + 1), receiveCount, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
        
		processWorking = 0;
        //He decidido emplear solo una region paralela para que los threads solo se lancen una vez
        #pragma omp parallel private(k)
        {
            for (k = 1; k <= dim; k++) {
                //Se determina a que proceso corresponde la columna que se esta tratando
                #pragma omp single
                {
                    if (processWorking != size - 1 && k > sendFrom[processWorking + 1]) {
                        processWorking++;
                    }
                }

                //El proceso al que corresponde la columna, calcula el máximo
                piv = -1; iPiv = -1;
                double piv_local = dabs(Aref(k, k));
				int iPiv_local = k;
				double currVal; 
                if (processWorking == myId) {
                    #pragma omp for
                    for (i = k + 1; i <= m; i++) {
                        currVal = dabs(Aref(i, k));
                        if (piv_local < currVal) {
                            piv_local = currVal;
                            iPiv_local = i;
                        }
                    }

                    #pragma omp critical
                    {
                        if (piv_local > piv) {
                            piv = piv_local;
                            iPiv = iPiv_local;
                        }
                    }

					#pragma omp barrier //Barrier para que todos tengan el iPiv correcto

                    #pragma omp master
                    {
                        dref(k) = iPiv;
                        piv = Aref(iPiv, k);
                    }
                }			

                //Se distribuye el indice (fila) donde está el máximo
                #pragma omp barrier
				#pragma omp master
				{
					MPI_Bcast(&iPiv, 1, MPI_INT, processWorking, MPI_COMM_WORLD);
				}
				#pragma omp barrier

                if (iPiv != k) {
                    #pragma omp for
                    for (i = startFrom + 1; i <= startFrom + receiveCount; i++) {
                        vtemp[i - 1] = Aref(k, i);
                        Aref(k, i) = Aref(iPiv, i);
                        Aref(iPiv, i) = vtemp[i - 1];
                    }
            
                    #pragma omp master
                    {
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
                                double ptmp = bref(k);
                                bref(k) = bref(iPiv);
                                bref(iPiv) = ptmp;
                        } else if (bPivLocation == myId){
                            MPI_Sendrecv_replace(&bref(iPiv), 1, MPI_DOUBLE, processWorking, TAG, processWorking, TAG, MPI_COMM_WORLD, &st);
                        } else if (processWorking == myId) {
                            MPI_Sendrecv_replace(&bref(k), 1, MPI_DOUBLE, bPivLocation, TAG, bPivLocation, TAG, MPI_COMM_WORLD, &st);
                        }
                    }
                }
        
                if (processWorking == myId) {
                    #pragma omp for
                    for (i = k + 1; i <= m; i++) {
                        Aref(i, k) /= piv;
                    }
                }

                //Se distribuye la columna y el elemento del vector
                #pragma omp barrier
                #pragma omp master
                {
					//MPI_Bcast(&Aref(1, k), 1, colType, processWorking, MPI_COMM_WORLD);
					//MPI_Bcast(&bref(k), 1, MPI_DOUBLE, processWorking, MPI_COMM_WORLD);
                    //Enviamos Aref(1,k) y bref(k) a los procesos que lo van a necesitar (id > k)
                    //Tambien se podría hacer Bcast, pero no todos los procesos lo necesitan
                    if (processWorking == myId){
                        for (i = myId + 1; i < size; i++) {
                            MPI_Send(&Aref(1, k), 1, colType, i, TAG + 1, MPI_COMM_WORLD);
                            MPI_Send(&bref(k), 1, MPI_DOUBLE, i, TAG, MPI_COMM_WORLD);
                        }
                    } else if (processWorking < myId) {
                        MPI_Recv(&Aref(1, k), 1, colType, processWorking, TAG + 1, MPI_COMM_WORLD, &st);
                        MPI_Recv(&bref(k), 1, MPI_DOUBLE, processWorking, TAG, MPI_COMM_WORLD, &st);
                    }
                }
                #pragma omp barrier      

                //Cada proceso realiza el calculo en sus columnas (limite inferior --> if, superior --> cond. segundo for)
                #pragma omp for private(j) schedule(dynamic)
                for (i = k + 1; i <= m; i++) {
                    for (j = k + 1; j <= startFrom + receiveCount; j++) {
                        if (j > startFrom){
                            Aref(i, j) -= Aref(i, k) * Aref(k, j);
                        }
                    }
                    //Actualiza b(i) si lo tiene
                    if (i > startFrom && i <= startFrom + receiveCount){
                        bref(i) -= Aref(i, k) * bref(k);
                    }
                }
            }

            #pragma omp barrier
			#pragma omp master //El tiempo solo toma un thread del proceso
			{
                MPI_Barrier(MPI_COMM_WORLD); //El proceso espera a los otros
				t2 = MPI_Wtime();
				timeLU += (t2 > t1 ? t2 - t1 : 0.0);
			}


			#pragma omp barrier
			#pragma omp master
			{
                MPI_Barrier(MPI_COMM_WORLD);
				t1 = MPI_Wtime();
			}

            //Backward substitution

            //He tenido que modificar algo el codigo del algoritmo, ya que he supuesto que 
            //cada proceso tiene solo sus bref (cosa que no es cierta, algunos tienen los bref de otros procesos)
            //Ej: todos tienen los bref del proceso 0, porque este se los envia para hacer la LU
            //Si el envio de los bref antes de la LU se hiciese con Bcast, esta parte la podria realizar
            //solo un proceso, pero me ha parecido más interesente suponer que un proceso solo tiene sus bref
            processWorkingA = size - 1;
            partialResultsToReceive = 0;
            for (k = dim; k > 0; k--) {
                //Se determina que proceso tiene el elemento k del vector 
                #pragma omp single
                {
                    if (processWorkingA != 0 && k == sendFrom[processWorkingA]) {
                        processWorkingA--;
                    }

                    if (processWorkingA == myId) {
                        partialResultsToReceive = size - 1 - processWorkingA;
                    }
                }
                
                //Si processWorkingA != myId, los threads se guardan localmente la cantidad acumulada que deben restar a b(k)
                //Posteriormente, los threads del mismo proceso acumulan estas cantidades, y el master envia esta cantidad
                //al proceso al cual pertenece b(k)
                double partialResultThread = 0.0;
                partialResultProcess = 0.0;
                #pragma omp for private(j) schedule(dynamic)
                for (i = k + 1; i <= n; i++) {                    
                    int processWorkingB = 0;
                    for (j = 1; j < size; j++) {
                        if (i > sendFrom[j] && i <= sendFrom[j] + sizeToSend[j]) {
                            processWorkingB = j;
                            break;
                        }
                    }
                             
                    if (processWorkingA != processWorkingB) {
                        if (processWorkingB == myId) {
                            partialResultThread += bref(i) * Aref(k, i);
                        }
                    } else {
                        if (processWorkingB == myId) {
                            bref(k) -= bref(i) * Aref(k, i);
                        }
                    } 
                }

                if (processWorkingA != myId){
                    #pragma omp critical
                    {
                        partialResultProcess += partialResultThread;
                    }
                }

                #pragma omp barrier
                #pragma omp master
                {
                    if (processWorkingA == myId){
                        for (i = 0; i < partialResultsToReceive; i++) {
                            MPI_Recv(&partialResultProcess, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
                            bref(k) -= partialResultProcess;
                        }
                    } else {
                        MPI_Send(&partialResultProcess, 1, MPI_DOUBLE, processWorkingA, TAG, MPI_COMM_WORLD);
                    }
                }
                #pragma omp barrier

                #pragma omp single
                {
                    if (processWorkingA == myId) {		
                        bref(k) /= Aref(k, k);
                    }
                }
            }
    
            //Obtain the solution
            #pragma omp for
            for (i = 1; i <= dim; i++) {
                xref(i) = bref(i);
            }
        }
   
        //Root recoge los resultados parciales para comprobar el correcto resultado
        MPI_Gatherv(myId == ROOT ? MPI_IN_PLACE : &xref(startFrom + 1), receiveCount, MPI_DOUBLE, x, sizeToSend, sendFrom, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

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
