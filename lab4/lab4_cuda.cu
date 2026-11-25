#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

__global__ void matMulNaive(const double* A, const double* B, double* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

template <int TILE>
__global__ void matMulTiled(const double* A, const double* B, double* C, int N) {
    __shared__ double As[TILE][TILE];
    __shared__ double Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    double sum = 0.0;

    int numTiles = (N + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; ++t) {
        int aRow = row;
        int aCol = t * TILE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] =
            (aRow < N && aCol < N) ? A[aRow * N + aCol] : 0.0;

        int bRow = t * TILE + threadIdx.y;
        int bCol = col;
        Bs[threadIdx.y][threadIdx.x] =
            (bRow < N && bCol < N) ? B[bRow * N + bCol] : 0.0;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

double* readMatrix(const char* filename, int* n) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Couldn't open file: %s\n", filename);
        exit(EXIT_FAILURE);
    }
    if (fscanf(file, "%d", n) != 1 || *n <= 0) {
        fprintf(stderr, "Error: Failed to read matrix size from: %s\n", filename);
        fclose(file);
        exit(EXIT_FAILURE);
    }

    size_t total = (size_t)(*n) * (*n);
    double* matrix = (double*)malloc(total * sizeof(double));
    if (!matrix) {
        fprintf(stderr, "Error: Host memory allocation failed\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < total; ++i) {
        if (fscanf(file, "%lf", &matrix[i]) != 1) {
            fprintf(stderr, "Error: Invalid data in: %s\n", filename);
            free(matrix);
            fclose(file);
            exit(EXIT_FAILURE);
        }
    }
    fclose(file);
    return matrix;
}

int main(int argc, char** argv) {
    const char* fileA = "matrixA.txt";
    const char* fileB = "matrixB.txt";
    const char* fileResult = "result.txt";

    const char* kernelType = "tiled";
    int blockDimX = 16;
    int blockDimY = 16;

    if (argc > 1) kernelType = argv[1];
    if (argc > 2) blockDimX = atoi(argv[2]);
    if (argc > 3) blockDimY = atoi(argv[3]);

    if (strcmp(kernelType, "naive") != 0 && strcmp(kernelType, "tiled") != 0) {
        fprintf(stderr, "Unknown kernel type: %s (use 'naive' or 'tiled')\n", kernelType);
        return 1;
    }

    int nA = 0, nB = 0;
    double* A = readMatrix(fileA, &nA);
    double* B = readMatrix(fileB, &nB);

    if (nA != nB) {
        fprintf(stderr, "Error: Matrices must have same dimensions!\n");
        free(A);
        free(B);
        return 1;
    }

    int N = nA;
    size_t totalElements = (size_t)N * N;
    size_t bytes = totalElements * sizeof(double);
    double* C = (double*)calloc(totalElements, sizeof(double));

    double *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice));

    dim3 block(blockDimX, blockDimY);
    dim3 grid((N + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);

    bool isNaive = (strcmp(kernelType, "naive") == 0);

    if (isNaive) {
        matMulNaive<<<grid, block>>>(d_A, d_B, d_C, N);
    } else {
        switch (blockDimX) {
            case 8:  matMulTiled<8><<<grid, block>>>(d_A, d_B, d_C, N);  break;
            case 16: matMulTiled<16><<<grid, block>>>(d_A, d_B, d_C, N); break;
            case 32: matMulTiled<32><<<grid, block>>>(d_A, d_B, d_C, N); break;
            default: matMulTiled<16><<<grid, block>>>(d_A, d_B, d_C, N); break;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    if (isNaive) {
        matMulNaive<<<grid, block>>>(d_A, d_B, d_C, N);
    } else {
        switch (blockDimX) {
            case 8:  matMulTiled<8><<<grid, block>>>(d_A, d_B, d_C, N);  break;
            case 16: matMulTiled<16><<<grid, block>>>(d_A, d_B, d_C, N); break;
            case 32: matMulTiled<32><<<grid, block>>>(d_A, d_B, d_C, N); break;
            default:
                fprintf(stderr, "TILE must be 8, 16 or 32 (blockDimX). Using 16.\n");
                matMulTiled<16><<<grid, block>>>(d_A, d_B, d_C, N);
                break;
        }
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    FILE* outFile = fopen(fileResult, "w");
    if (!outFile) {
        fprintf(stderr, "Error: Could not create output file: %s\n", fileResult);
        free(A); free(B); free(C);
        return 1;
    }

    fprintf(outFile, "Execution stats ->\n");
    fprintf(outFile, "Kernel                     : %s\n", kernelType);
    fprintf(outFile, "Block dimensions           : %d x %d\n", blockDimX, blockDimY);
    fprintf(outFile, "Grid dimensions            : %u x %u\n", grid.x, grid.y);
    fprintf(outFile, "Problem Size (Dimension N) : %d x %d\n", N, N);
    fprintf(outFile, "Total Elements per Matrix  : %zu\n", totalElements);
    fprintf(outFile, "Number of Multiplications  : %zu\n", totalElements * N);
    fprintf(outFile, "Execution Time (kernel)    : %.4f ms\n\n\n", milliseconds);

    fprintf(outFile, "Result Matrix C:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            fprintf(outFile, "%10.2f ", C[i * N + j]);
        }
        fprintf(outFile, "\n");
    }
    fclose(outFile);

    printf("CUDA calculation completed. Kernel=%s, block=%dx%d, grid=%ux%u, time=%.4f ms. Results saved to %s\n",
           kernelType, blockDimX, blockDimY, grid.x, grid.y, milliseconds, fileResult);

    free(A);
    free(B);
    free(C);

    printf("Running automated verification!\n");
    int returnCode = system("py verify.py");
    if (returnCode != 0) {
        printf("Warning: Python verification script returned an error.\n");
    }

    return 0;
}