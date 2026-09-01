
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <algorithm>

// CUDA kernel for matrix multiplication
__global__ void matrixMultiplyGPU(
    const float *A,
    const float *B,
    float *C,
    int N)
{
    // Find the row and column handled by this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure the thread is inside the matrix
    if (row < N && col < N)
    {
        float sum = 0.0f;

        // Multiply one row of A with one column of B
        for (int k = 0; k < N; k++)
        {
            sum += A[row * N + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}


// CPU version of matrix multiplication
void matrixMultiplyCPU(
    const float *A,
    const float *B,
    float *C,
    int N)
{
    // Start the output matrix with zeros
    std::fill(C, C + (long long)N * N, 0.0f);

    for (int i = 0; i < N; i++)
    {
        for (int k = 0; k < N; k++)
        {
            float a = A[i * N + k];

            for (int j = 0; j < N; j++)
            {
                C[i * N + j] += a * B[k * N + j];
            }
        }
    }
}


// Fill matrices with deterministic values
void initializeMatrix(float *matrix, int N, int offset)
{
    long long total = (long long)N * N;

    for (long long i = 0; i < total; i++)
    {
        matrix[i] = ((i + offset) % 100) / 100.0f;
    }
}


// Run the experiment for one matrix size
void runExperiment(int N)
{
    printf("\n========================================\n");
    printf("Matrix size: %d x %d\n", N, N);
    printf("========================================\n");

    long long elements = (long long)N * N;
    size_t bytes = elements * sizeof(float);

    // -----------------------------
    // Allocate CPU memory
    // -----------------------------
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C_CPU = (float *)malloc(bytes);
    float *h_C_GPU = (float *)malloc(bytes);

    initializeMatrix(h_A, N, 1);
    initializeMatrix(h_B, N, 7);

    // -----------------------------
    // CPU timing
    // -----------------------------
    auto cpu_start = std::chrono::high_resolution_clock::now();

    matrixMultiplyCPU(h_A, h_B, h_C_CPU, N);

    auto cpu_end = std::chrono::high_resolution_clock::now();

    double cpu_time =
        std::chrono::duration<double, std::milli>(
            cpu_end - cpu_start
        ).count();


    // -----------------------------
    // Allocate GPU memory
    // -----------------------------
    float *d_A, *d_B, *d_C;

    cudaMalloc((void **)&d_A, bytes);
    cudaMalloc((void **)&d_B, bytes);
    cudaMalloc((void **)&d_C, bytes);


    // CUDA events used for timing
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    // -----------------------------
    // Host-to-Device transfer timing
    // -----------------------------
    cudaEventRecord(start);

    cudaMemcpy(
        d_A,
        h_A,
        bytes,
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_B,
        h_B,
        bytes,
        cudaMemcpyHostToDevice
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float h2d_time = 0.0f;

    cudaEventElapsedTime(
        &h2d_time,
        start,
        stop
    );


    // -----------------------------
    // CUDA block and grid setup
    // -----------------------------
    dim3 block(16, 16);

    dim3 grid(
        (N + block.x - 1) / block.x,
        (N + block.y - 1) / block.y
    );


    // -----------------------------
    // Warm-up GPU kernel
    // -----------------------------
    matrixMultiplyGPU<<<grid, block>>>(
        d_A,
        d_B,
        d_C,
        N
    );

    cudaDeviceSynchronize();


    // -----------------------------
    // GPU kernel timing
    // -----------------------------
    cudaEventRecord(start);

    matrixMultiplyGPU<<<grid, block>>>(
        d_A,
        d_B,
        d_C,
        N
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float kernel_time = 0.0f;

    cudaEventElapsedTime(
        &kernel_time,
        start,
        stop
    );


    // -----------------------------
    // Device-to-Host transfer timing
    // -----------------------------
    cudaEventRecord(start);

    cudaMemcpy(
        h_C_GPU,
        d_C,
        bytes,
        cudaMemcpyDeviceToHost
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float d2h_time = 0.0f;

    cudaEventElapsedTime(
        &d2h_time,
        start,
        stop
    );


    // -----------------------------
    // Correctness check
    // -----------------------------
    float max_error = 0.0f;

    for (long long i = 0; i < elements; i++)
    {
        float error =
            fabs(h_C_CPU[i] - h_C_GPU[i]);

        if (error > max_error)
        {
            max_error = error;
        }
    }


    // -----------------------------
    // Final timing calculations
    // -----------------------------
    float transfer_time =
        h2d_time + d2h_time;

    float gpu_end_to_end =
        transfer_time + kernel_time;

    double speedup =
        cpu_time / gpu_end_to_end;


    // -----------------------------
    // Print results
    // -----------------------------
    printf("CPU time: %.3f ms\n", cpu_time);

    printf(
        "GPU kernel time: %.3f ms\n",
        kernel_time
    );

    printf(
        "H2D time: %.3f ms\n",
        h2d_time
    );

    printf(
        "D2H time: %.3f ms\n",
        d2h_time
    );

    printf(
        "H2D + D2H time: %.3f ms\n",
        transfer_time
    );

    printf(
        "GPU end-to-end time: %.3f ms\n",
        gpu_end_to_end
    );

    printf(
        "End-to-end speedup: %.3fx\n",
        speedup
    );

    printf(
        "Maximum error: %.6f\n",
        max_error
    );


    // -----------------------------
    // Cleanup
    // -----------------------------
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C_CPU);
    free(h_C_GPU);
}


int main(int argc, char **argv)
{
    // If a size is supplied, run only that size.
    // This will be useful later for profiling.
    if (argc == 2)
    {
        int N = atoi(argv[1]);
        runExperiment(N);
        return 0;
    }

    // Required matrix sizes for HW1
    int sizes[] = {
        256,
        1024,
        4096
    };

    for (int i = 0; i < 3; i++)
    {
        runExperiment(sizes[i]);
    }

    return 0;
}
