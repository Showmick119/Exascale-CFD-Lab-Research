#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>

void matrix_addition(int N, const std::vector<std::vector<int>>& A, const std::vector<std::vector<int>>& B, std::vector<std::vector<int>>& C) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
     auto end  = std::chrono::high_resolution_clock::now();
     std::chrono::duration<double> elapsed = end - start;
     std::cout <<"Matrix size: " <<N << " Execution time: " <<elapsed.count() << " seconds" <<std::endl;
}

int main() {
    int N = 100;

    std::vector<std::vector<int>> A(N, std::vector<int>(N, 1));
    std::vector<std::vector<int>> B(N, std::vector<int>(N, 2));
    std::vector<std::vector<int>> C(N, std::vector<int>(N, 0));

    matrix_addition(N, A, B, C);
}
