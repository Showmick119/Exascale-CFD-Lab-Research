#include <Kokkos_Core.hpp>
#include <Kokkos_Timer.hpp>
#include <cuda_runtime.h>
#include <omp.h>
#include <iostream>
#include <iomanip>

int getNumGPUs() {
    int numGPUs = 0;
    cudaGetDeviceCount(&numGPUs);
    return numGPUs;
}

double matrixAddMultiGPU(const int N, const int numGPUs) {
    Kokkos::Timer timer;

    #pragma omp parallel num_threads(numGPUs)
    {
        int gpu = omp_get_thread_num();

        cudaSetDevice(gpu);

        int rowsPerGPU = N / numGPUs;
        int remainder = N % numGPUs;

        int startRow = gpu * rowsPerGPU;
        int endRow = startRow + rowsPerGPU;
        if (gpu == numGPUs - 1) {
            endRow += remainder;
        }

        Kokkos::View<double**> A("A", N, N);
        Kokkos::View<double**> B("B", N, N);
        Kokkos::View<double**> C("C", N, N);

        Kokkos::parallel_for("Initialize A and B", Kokkos::RangePolicy<>(startRow, endRow), KOKKOS_LAMBDA(const int i) {
            for (int j = 0; j < N; ++j) {
                A(i, j) = i * N + j;
                B(i, j) = (i * N + j) * 0.5;
            }
        });

        Kokkos::parallel_for("Matrix Addition", Kokkos::RangePolicy<>(startRow, endRow), KOKKOS_LAMBDA(const int i) {
            for (int j = 0; j < N; ++j) {
                C(i, j) = A(i, j) + B(i, j);
            }
        });

        Kokkos::fence();

        #pragma omp critical
        {
            std::cout << "GPU " << gpu << " completed computation on rows " << startRow
                      << " to " << endRow - 1 << std::endl;
        }
    }

    return timer.seconds();
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);

    {
        int N = 3500;
        int numGPUs = getNumGPUs();

        std::cout << "Matrix size: " << N << "x" << N << std::endl;
        std::cout << "Number of GPUs: " << numGPUs << std::endl;

        double elapsed = matrixAddMultiGPU(N, numGPUs);

        std::cout << "Time taken for matrix addition with size " << N << "x" << N
                  << " using " << numGPUs << " GPUs: " << elapsed << " seconds." << std::endl;
    }

    Kokkos::finalize();
    return 0;
}
