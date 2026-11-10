#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4) {
        if (rank == 0) printf("Использование: %s <N> <A.txt> <B.txt>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int N = atoi(argv[1]);
    const char* fileA = argv[2];
    const char* fileB = argv[3];

    double *A = NULL, *B = NULL, *C = NULL;
    
    if (rank == 0) {
        A = (double*)malloc(N * N * sizeof(double));
        B = (double*)malloc(N * N * sizeof(double));
        C = (double*)malloc(N * N * sizeof(double));
    } else {
        B = (double*)malloc(N * N * sizeof(double));
    }

    if (rank == 0) {
        FILE *fa = fopen(fileA, "r");
        FILE *fb = fopen(fileB, "r");
        for (int i = 0; i < N * N; i++) fscanf(fa, "%lf", &A[i]);
        for (int i = 0; i < N * N; i++) fscanf(fb, "%lf", &B[i]);
        fclose(fa); fclose(fb);
    }

    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int rows_per_proc = N / size;
    double *local_A = (double*)malloc(rows_per_proc * N * sizeof(double));
    double *local_C = (double*)malloc(rows_per_proc * N * sizeof(double));

    MPI_Scatter(A, rows_per_proc * N, MPI_DOUBLE, local_A, rows_per_proc * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double start_time = MPI_Wtime();

    for (int i = 0; i < rows_per_proc; i++) {
        for (int j = 0; j < N; j++) {
            local_C[i * N + j] = 0.0;
            for (int k = 0; k < N; k++) {
                local_C[i * N + j] += local_A[i * N + k] * B[k * N + j];
            }
        }
    }

    MPI_Gather(local_C, rows_per_proc * N, MPI_DOUBLE, C, rows_per_proc * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Time: %f sec | Processes: %d | N: %d\n", end_time - start_time, size, N);
        FILE *fc = fopen("result.txt", "w");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) fprintf(fc, "%.4f ", C[i * N + j]);
            fprintf(fc, "\n");
        }
        fclose(fc);
        free(A); free(B); free(C);
    }

    free(local_A); free(local_C);
    if (rank != 0) free(B);
    MPI_Finalize();
    return 0;
}