#include <iostream>
#include <vector>
#include <chrono>

void matVecMult(int N, const std::vector<std::vector<double>>& A, const std::vector<double>& V, std::vector<double>& C) {
    for (int i = 0; i < N; ++i) {
        C[i] = 0.0;
        for (int j = 0; j < N; ++j) {
            C[i] += A[i][j] * V[j];
        }
    }
}

int main() {
    int N = 1000;
    
    std::vector<std::vector<double>> A(N, std::vector<double>(N, 1.0));
    std::vector<double> V(N, 1.0);
    std::vector<double> C(N, 0.0);

    auto start = std::chrono::high_resolution_clock::now();

    matVecMult(N, A, V, C);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    std::cout << "Matrix size: " << N << "x" << N << "\n";
    std::cout << "Execution time: " << elapsed.count() << " seconds\n";
    
    return 0;
}
