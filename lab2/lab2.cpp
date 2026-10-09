#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <stdexcept>
#include <omp.h>

using namespace std;

void readmatrix(const string& filename, vector<vector<double>>& matrix, int& n) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Couldn't open file: " + filename);
    }

    if (!(file >> n) || n <= 0) {
        throw runtime_error("Failed to read matrix size from: " + filename);
    }

    matrix.assign(n, vector<double>(n));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (!(file >> matrix[i][j])) {
                throw runtime_error("Invalid matrix file content: " + filename);
            }
        }
    }
    file.close();
}

int main(int argc, char* argv[]) {
    int num_threads = 4; 
    if (argc > 1) {
        num_threads = atoi(argv[1]);
    }
    omp_set_num_threads(num_threads);

    string fileA = "matrixA.txt";
    string fileB = "matrixB.txt";
    string fileResult = "result.txt";

    int nA = 0, nB = 0;
    vector<vector<double>> A, B;
    
    try {
        readmatrix(fileA, A, nA);
        readmatrix(fileB, B, nB);
    } catch (const exception& e) {
        cerr << "Initialization error (waiting for matrix generation): " << e.what() << "\n";
        return 1;
    }

    if (nA != nB) {
        cerr << "Error: Matrices must have same dimensions!\n";
        return 1;
    }

    int N = nA;
    vector<vector<double>> C(N, vector<double>(N, 0.0));

    double start = omp_get_wtime();

    #pragma omp parallel for shared(A, B, C, N) schedule(static)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }

    double end = omp_get_wtime();
    double duration_ms = (end - start) * 1000.0;

    ofstream outFile(fileResult);
    if (outFile.is_open()) {
        outFile << "Execution stats ->\n";
        outFile << "Problem Size (Dimension N) : " << N << " x " << N << "\n";
        outFile << "Threads Used               : " << num_threads << "\n";
        outFile << "Execution Time             : " << fixed << setprecision(4) << duration_ms << " ms\n";
        outFile.close();
    }

    cout << "Success\n";
    return 0;
}
