#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <stdexcept>

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
                throw runtime_error("Invalid: " + filename);
            }
        }
    }
    file.close();
}

int main() {
    string fileA = "matrixA.txt";
    string fileB = "matrixB.txt";
    string fileResult = "result.txt";

    int nA = 0, nB = 0;
    vector<vector<double>> A, B;
    readmatrix(fileA, A, nA);
    readmatrix(fileB, B, nB);

    if (nA != nB) {
        throw invalid_argument("Matrices must have same dimensions!");
    }

    int N = nA;
    vector<vector<double>> C(N, vector<double>(N, 0.0));

    auto start = chrono::high_resolution_clock::now();

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration = end - start;

    ofstream outFile(fileResult);
    if (!outFile.is_open()) {
        throw runtime_error("Could not create output file: " + fileResult);
    }

    outFile << "Execution stats ->\n";
    outFile << "Problem Size (Dimension N) : " << N << " x " << N << "\n";
    outFile << "Total Elements per Matrix  : " << N * N << "\n";
    outFile << "Number of Multiplications  : " << N * N * N << "\n";
    outFile << "Execution Time             : " << fixed << setprecision(4) << duration.count() << " ms\n";
    outFile << "\n\n";

    outFile << "Result Matrix C:\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            outFile << setw(10) << fixed << setprecision(2) << C[i][j] << " ";
        }
        outFile << "\n";
    }
    outFile.close();

    cout << "Calculation completed successfully. Results saved to " << fileResult << "\n";

    cout << "Running automated verification!\n";
    int returnCode = system("py verify.py");
    if (returnCode != 0) {
        cout << "Warning: Python verification script returned an error.\n";
    }

    return 0;
}