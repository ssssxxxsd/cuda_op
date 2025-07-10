
#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>

#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)


#define CHECK_CUBLAS(call) { \
    cublasStatus_t stat = call; \
    if (stat != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, stat); \
        exit(EXIT_FAILURE); \
    } \
}

/// Kernel to initialize a matrix with small integers.
template <typename T>
__global__ void InitializeMatrix_kernel(
  T *matrix,
  int rows,
  int columns,
  int seed = 0) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < rows && j < columns) {
    int offset = i + j * rows;

    // Generate arbitrary elements.
    int const k = 16807;
    int const m = 16;
    matrix[offset] = T(((offset + seed) * k % m) - m / 2);
  }
}

template <typename T>
__global__ void InitializeMatrixOrder_kernel(
  T *matrix,
  int rows,
  int columns) {

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      int offset = i * columns + j;
      matrix[offset] = T(i + j + 1.0);
    }
  }
}

/// Simple function to initialize a matrix to arbitrary small integers.
template <typename T>
cudaError_t InitializeMatrix(T *matrix, int rows, int columns, int seed = 0) {

  dim3 block(16, 16);
  dim3 grid(
    (rows + block.x - 1) / block.x,
    (columns + block.y - 1) / block.y
  );

  if(seed < 0) {
    InitializeMatrixOrder_kernel<<< 1, 1 >>>(matrix, rows, columns);
  } else {
    InitializeMatrix_kernel<<< grid, block >>>(matrix, rows, columns, seed);
  }
  return cudaGetLastError();
}



/// Allocates device memory for a matrix then fills with arbitrary small integers.
template <typename T>
cudaError_t AllocateMatrix(T **matrix, int rows, int columns, int seed = 0, bool needInit = true) {
  cudaError_t result;

  size_t sizeof_matrix = sizeof(T) * rows * columns;

  // Allocate device memory.
  result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Clear the allocation.
  result = cudaMemset(*matrix, 0, sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to clear matrix device memory: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Initialize matrix elements to arbitrary small integers.
  if (!needInit) {
    return result;
  }


  result = InitializeMatrix(*matrix, rows, columns, seed);
  
  if (result != cudaSuccess) {
    std::cerr << "Failed to initialize matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  return result;
}


template <typename T>
__device__ void PrintMatrix_device(
  T *matrix,
  int rows,
  int columns) {

  for (int i = 0; i < rows; i++) {
    printf("%3d: ", i);
    for (int j = 0; j < columns; j++) {
      int offset = i * columns + j;
      printf("%8.0f ", float(matrix[offset]));
    }
    printf("\n");
  }
}

// 设备端 kernel：将数组中每个元素设置为指定值
__device__ void deviceMemset(void* ptr, int value, size_t N) {
    for (size_t i = 0; i < N; i++) {
        ((unsigned char*)ptr)[i] = value;
    }
}


template <typename T>
__global__ void PrintMatrix_kernel(
  T *matrix,
  int rows,
  int columns) {

  for (int i = 0; i < rows; i++) {
    printf("%3d: ", i);
    for (int j = 0; j < columns; j++) {
      int offset = i * columns + j;
      printf("%8.0f ", float(matrix[offset]));
    }
    printf("\n");
  }
}


template <typename T>
cudaError_t PrintMatrix(T *matrix, int rows, int columns) {
  PrintMatrix_kernel<<< 1, 1 >>>(matrix, rows, columns);
  return cudaGetLastError();
}

template <typename T>
__global__ void MatrixCompareKernel(T *matrixA, T *matrixB, int rows, int columns) {
  int diffCount = 0;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      int offset = i * columns + j;
      if (matrixA[offset] != matrixB[offset]) {
        diffCount++;
        if(diffCount < 10) {
          printf("Diff matrixA[%d][%d] = %f, matrixB[%d][%d] = %f\n", i, j, matrixA[offset], i, j, matrixB[offset]);
        }
      }
    }
  }
  if(diffCount > 0) {
    printf("Matrix compare Fail Diff count: %d\n", diffCount);
  } else {
    printf("Matrix compare success Pass!\n");
  }
}


template <typename T>
cudaError_t MatrixCompare(T *matrixA, T *matrixB, int rows, int columns) {
  MatrixCompareKernel<<< 1, 1 >>>(matrixA, matrixB, rows, columns);
  return cudaGetLastError();
}

template <typename T, typename O>
__global__ void SimpleGemmKernel(T *A, T *B, O *C, int M, int N, int K) {

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      O sum = 0;
      for (int k = 0; k < K; k++) {
        sum += float(A[i * K + k]) * float(B[j * K + k]);
      }
      C[i * N + j] = sum;
    }
  }

}


template <typename T, typename O>
cudaError_t SimpleGemm(T *A, T *B, O *C, int M, int N, int K) {
  SimpleGemmKernel<<< 1, 1 >>>(A, B, C, M, N, K);
  return cudaGetLastError();
}


//编译器log函数
template <unsigned int N>
struct ConstExprLog2 {
    static_assert(N > 0, "Log2 is undefined for 0");
    static_assert(N % 2 == 0, "Log2 is undefined for odd numbers");
    static constexpr unsigned int value = 1 + ConstExprLog2<N / 2>::value;
};
// 递归终止条件：N == 1 时 log2(1) = 0
template <>
struct ConstExprLog2<1> {
    static constexpr unsigned int value = 0;
};


__forceinline__ __device__ __host__ void print_binary(uint32_t n) {
    int bits = sizeof(n) * 8;  // 计算位数，比如32位
    for (int i = bits - 1; i >= 0; i--) {
        unsigned int bit = (n >> i) & 1;
        printf("%u", bit);
    }
}