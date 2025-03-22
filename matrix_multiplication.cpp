#include <iostream>
#include <vector>
#include <chrono>

// Matrix-Vector Multiplication Kernel
void matVecMult(int N, const std::vector<std::vector<double>>& A, const std::vector<double>& V, std::vector<double>& C) {
    for (int i = 0; i < N; ++i) {
        C[i] = 0.0; // Ensure output vector is zeroed before accumulation
        for (int j = 0; j < N; ++j) {
            C[i] += A[i][j] * V[j];
        }
    }
}

int main() {
    int N = 1000; // Example size, can be modified
    
    // Allocate and initialize input matrix and vector
    std::vector<std::vector<double>> A(N, std::vector<double>(N, 1.0)); // Example: All elements set to 1.0
    std::vector<double> V(N, 1.0); // Example: All elements set to 1.0
    std::vector<double> C(N, 0.0); // Output vector

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();
    
    // Execute matrix-vector multiplication
    matVecMult(N, A, V, C);
    
    // End timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    // Print results
    std::cout << "Matrix size: " << N << "x" << N << "\n";
    std::cout << "Execution time: " << elapsed.count() << " seconds\n";
    
    return 0;
}