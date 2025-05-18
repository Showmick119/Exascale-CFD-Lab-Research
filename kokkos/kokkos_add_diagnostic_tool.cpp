#include <Kokkos_Core.hpp>
#include <cuda_runtime.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <vector>

struct GPUData {
    std::vector<int> matrix_sizes;
    std::vector<double> host_to_device_times;
    std::vector<double> computation_times;
    std::vector<double> device_to_host_times;
    std::vector<double> synchronization_times;
};

void measureOverheadsMultiGPU(int N, int numGPUs, std::vector<GPUData>& gpuData) {
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

        Kokkos::View<double**, Kokkos::CudaUVMSpace> A("A", N, N);
        Kokkos::View<double**, Kokkos::CudaUVMSpace> B("B", N, N);
        Kokkos::View<double**, Kokkos::CudaUVMSpace> C("C", N, N);

        const int chunkSize = 5000;
        Kokkos::Timer transfer_to_device_timer;
        for (int i = 0; i < N; i += chunkSize) {
            int chunkEnd = std::min(i + chunkSize, N);
            auto A_chunk = Kokkos::subview(A, std::make_pair(i, chunkEnd), Kokkos::ALL());
            Kokkos::deep_copy(A_chunk, A_chunk);
        }
        double transfer_to_device_time = transfer_to_device_timer.seconds();

        Kokkos::Timer computation_timer;
        Kokkos::parallel_for("MatrixAdd",
                             Kokkos::TeamPolicy<>(endRow - startRow, Kokkos::AUTO),
                             KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
                                 int i = team.league_rank() + startRow;
                                 Kokkos::parallel_for(Kokkos::TeamThreadRange(team, N), [&](int j) {
                                     C(i, j) = A(i, j) + B(i, j);
                                 });
                             });
        Kokkos::fence();
        double computation_time = computation_timer.seconds();

        Kokkos::Timer transfer_to_host_timer;
        for (int i = 0; i < N; i += chunkSize) {
            int chunkEnd = std::min(i + chunkSize, N);
            auto C_chunk = Kokkos::subview(C, std::make_pair(i, chunkEnd), Kokkos::ALL());
            Kokkos::deep_copy(C_chunk, C_chunk);
        }
        double transfer_to_host_time = transfer_to_host_timer.seconds();

        Kokkos::Timer sync_timer;
        Kokkos::fence();
        double sync_time = sync_timer.seconds();

        #pragma omp critical
        {
            gpuData[gpu].matrix_sizes.push_back(N);
            gpuData[gpu].host_to_device_times.push_back(transfer_to_device_time);
            gpuData[gpu].computation_times.push_back(computation_time);
            gpuData[gpu].device_to_host_times.push_back(transfer_to_host_time);
            gpuData[gpu].synchronization_times.push_back(sync_time);
        }
    }
}

int main() {
    Kokkos::initialize();
    {
        int numGPUs = 0;
        cudaGetDeviceCount(&numGPUs);

        if (numGPUs == 0) {
            std::cerr << "No GPUs available. Exiting..." << std::endl;
            Kokkos::finalize();
            return -1;
        }

        std::cout << "Number of GPUs detected: " << numGPUs << std::endl;

        std::vector<GPUData> gpuData(numGPUs);

        for (int N = 200; N <= 25000; N += 200) {
            measureOverheadsMultiGPU(N, numGPUs, gpuData);
        }

        std::ofstream file("gpu_overhead_data.csv");
        file << "GPU,Matrix Size,Host-to-Device Transfer Time,Computation Time,Device-to-Host Transfer Time,Synchronization Time\n";

        for (int gpu = 0; gpu < numGPUs; ++gpu) {
            for (size_t i = 0; i < gpuData[gpu].matrix_sizes.size(); ++i) {
                file << gpu << ","
                     << gpuData[gpu].matrix_sizes[i] << ","
                     << gpuData[gpu].host_to_device_times[i] << ","
                     << gpuData[gpu].computation_times[i] << ","
                     << gpuData[gpu].device_to_host_times[i] << ","
                     << gpuData[gpu].synchronization_times[i] << "\n";
            }
        }

        file.close();
        std::cout << "Data written to gpu_overhead_data.csv" << std::endl;
    }
    Kokkos::finalize();
    return 0;
}
