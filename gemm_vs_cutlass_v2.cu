// nvcc -std=c++17 -arch=sm_80 --expt-relaxed-constexpr -I../cutlass/include -I../cutlass/tools/profiler/include -I../cutlass/tools/util/include -lcublas -o gemm_vs_cutlass_v2 gemm_vs_cutlass_v2.cu
//  nvcc --generate-line-info --expt-relaxed-constexpr  -std=c++17 -arch=sm_80 -I../cutlass/include -I../cutlass/tools/profiler/include -I../cutlass/tools/util/include -lcublas -o gemm_vs_cutlass_v2 gemm_vs_cutlass_v2.cu
// /root/package/clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04/bin/clang++ gemm_vs_cutlass.cu -o gemm_vs_cutlass_clang -std=c++17 --cuda-gpu-arch=sm_80 -I../cutlass/include -I../cutlass/tools/profiler/include -I../cutlass/tools/util/include -L/usr/local/cuda/lib64 -lcudart_static -lcublas -ldl -lrt -pthread
//--expt-relaxed-constexpr

#include <iostream>
#include <sstream>
#include <vector>
#include <stdio.h>
#include <type_traits>

#include "cutlass/platform/platform.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "cutlass/gemm/kernel/default_gemm_universal.h"
#include "cutlass/util/packed_stride.hpp"

#include "cuda_utils.h"

static int const WarpThreadNum = 32;

static int const PipelineStageNum = 4;

static int const kRowsEpiloguePerIteration = 8;


using ThreadBlockShape = cutlass::gemm::GemmShape<128, 128, 16>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

using MmaIterations = cutlass::gemm::GemmShape<WarpShape::kM / InstructionShape::kM,
                                            WarpShape::kN / InstructionShape::kN,
                                            WarpShape::kK / InstructionShape::kK>;


static int const InstructionFragNumA = 4;
static int const InstructionFragNumB = 2;
static int const InstructionFragNumC = 4;
static int const InstructionFragWriteEleC = 2;


static int const InstructionReadCol = 4;
static int const InstructionReadRow = 8;
static int const InstructionReadThreadNumPerLine = 4; //指令级读取一行需要的线程数
static int const InstructionWriteRow = 8;
static int const InstructionWriteCol = 8;
static int const InstructionWriteThreadNumPerLine = 4;





static int const InstructionOnceReadLines = 8;

using WarpCount = cutlass::gemm::GemmShape<ThreadBlockShape::kM / WarpShape::kM,
                              ThreadBlockShape::kN / WarpShape::kN,
                              ThreadBlockShape::kK / WarpShape::kK>;

using ElementInput = cutlass::tfloat32_t;
//using ElementInput = float;

using ElementOutput = float;

// 每个线程读取的元素个数
static int const AccessInputElementSize = 16 / sizeof(ElementInput);


// 线程块读取的block行数
static_assert((AccessInputElementSize * WarpThreadNum * WarpCount::kCount) % ThreadBlockShape::kK == 0, 
    "线程块读取的数据大小必须是数据行数的整数倍");
static int const BlockLoadLine = (AccessInputElementSize * WarpThreadNum * WarpCount::kCount) / ThreadBlockShape::kK;

// warp读取的block行数
static_assert((AccessInputElementSize * WarpThreadNum) % ThreadBlockShape::kK == 0, 
    "warp读取的数据大小必须是数据行数的整数倍");
static int const WarpLoadLineA = (AccessInputElementSize * WarpThreadNum) / WarpShape::kK;

//block load一行需要的线程数
static_assert(ThreadBlockShape::kK % AccessInputElementSize == 0, 
    "block一行元素总数必须是 AccessInputElementSize 的整数倍");
static int const BlockLoadLineThreadNum = ThreadBlockShape::kK / AccessInputElementSize;


// 每个线程读取次数 = 读取的blockA 元素总数 / （每个线程一次读取的元素个数 * warp数 * warp内线程数）
static_assert(ThreadBlockShape::kMK % (AccessInputElementSize * WarpCount::kCount * WarpThreadNum) == 0, 
    "blockA 元素总数必须是 AccessInputElementSize * WarpCount::kCount * WarpThreadNum 的整数倍");
static int const BlockMemLoadIterNumA = ThreadBlockShape::kMK / (AccessInputElementSize * WarpCount::kCount * WarpThreadNum);

static_assert(ThreadBlockShape::kKN % (AccessInputElementSize * WarpCount::kCount * WarpThreadNum) == 0, 
    "blockB 元素总数必须是 AccessInputElementSize * WarpCount::kCount * WarpThreadNum 的整数倍");
static int const BlockMemLoadIterNumB = ThreadBlockShape::kKN / (AccessInputElementSize * WarpCount::kCount * WarpThreadNum);

template <int M = 2>
class Swizzle{
    static constexpr int S = ConstExprLog2<128 / sizeof(ElementInput)>::value - M;
    static constexpr int B = ConstExprLog2<InstructionOnceReadLines * ThreadBlockShape::kK>::value - M - S;
    static_assert(B >= 0, "Swizzle B must be greater than 0");
    static constexpr int MASK = ((1 << B) - 1) << (M + S);

    public:
      __device__ constexpr static int SwizzleOffset(const int& offset){
        return offset ^ ((offset & MASK) >> S);   // ZZZ ^= YYY
      }
};

using SwizzleLoad4Byte = Swizzle<2>;


struct Index{
    int blockAOffset;
    int blockBOffset;
    int blockDOffset;
    //int blockAWarpOffset;
    //int blockBWarpOffset;
    int tileSmemSaveSwizzleOffset;
    int blockAWarpSwizzleOffset[MmaIterations::kK];
    int blockBWarpSwizzleOffset[MmaIterations::kK];
    int kIterNum;



    __device__ Index(const int& M, const int& N, const int& K){
        assert(K % ThreadBlockShape::kK == 0);

        int warpidx = threadIdx.x / 32;
        int laneidx = threadIdx.x % 32;
        int threadidx = threadIdx.x;

        int warp_idx_mn = warpidx % (WarpCount::kM * WarpCount::kN);
        int warp_idx_k = warpidx / (WarpCount::kM * WarpCount::kN);

        int warp_idx_m = warp_idx_mn / WarpCount::kN;
        int warp_idx_n = warp_idx_mn % WarpCount::kN;

        int blockWarpOffset = warp_idx_m * WarpShape::kM * ThreadBlockShape::kK + warp_idx_k * WarpShape::kK;

        //thread/warp offset
        //blockAWarpOffset += laneidx / InstructionReadThreadNumPerLine * ThreadBlockShape::kK + laneidx % InstructionReadThreadNumPerLine;
        blockWarpOffset += (laneidx % InstructionShape::kM) * ThreadBlockShape::kK + laneidx / InstructionShape::kM * InstructionReadCol;


        for(int i = 0; i < MmaIterations::kK; ++i) {
            blockAWarpSwizzleOffset[i] = SwizzleLoad4Byte::SwizzleOffset(blockWarpOffset);
            blockWarpOffset += InstructionShape::kK;
        }

        blockWarpOffset = warp_idx_n * WarpShape::kN * ThreadBlockShape::kK + warp_idx_k * WarpShape::kK;
        //blockBWarpOffset += laneidx / InstructionReadThreadNumPerLine * ThreadBlockShape::kK + laneidx % InstructionReadThreadNumPerLine;
        blockWarpOffset += (laneidx % InstructionShape::kN) * ThreadBlockShape::kK + (laneidx % 16) / InstructionShape::kN * InstructionReadCol;

        for(int i = 0; i < MmaIterations::kK; ++i) {
            blockBWarpSwizzleOffset[i] = SwizzleLoad4Byte::SwizzleOffset(blockWarpOffset);
            blockWarpOffset += InstructionShape::kK;
        }

        blockAOffset = blockIdx.x * ThreadBlockShape::kM * K +  // block偏移
            (threadidx / BlockLoadLineThreadNum) * K +  // 行偏移
            (threadidx % BlockLoadLineThreadNum) * AccessInputElementSize;  // 列偏移

        blockBOffset = blockIdx.y * ThreadBlockShape::kN * K +  // block偏移
            (threadidx / BlockLoadLineThreadNum) * K +  // 行偏移
            (threadidx % BlockLoadLineThreadNum) * AccessInputElementSize;  // 列偏移

        blockDOffset = blockIdx.x * ThreadBlockShape::kM * N + blockIdx.y * ThreadBlockShape::kN +  // block 偏移
            warp_idx_m * WarpShape::kM * N + warp_idx_n * WarpShape::kN +  // warp 偏移
            laneidx / InstructionWriteThreadNumPerLine * N + laneidx % InstructionWriteThreadNumPerLine * InstructionFragWriteEleC;  // thread 偏移

        tileSmemSaveSwizzleOffset = (threadidx / BlockLoadLineThreadNum) * ThreadBlockShape::kK + (threadidx % BlockLoadLineThreadNum) * AccessInputElementSize;

        tileSmemSaveSwizzleOffset = SwizzleLoad4Byte::SwizzleOffset(tileSmemSaveSwizzleOffset);
    }

    /*
    static constexpr int swizzleTable[N + 1] = []() constexpr {
        int result[N + 1] = {1};  // 0! = 1
        for (int i = 1; i <= N; ++i) {
            result[i] = factorial(i);
        }
        return result;
    }();
    */

};



struct MmaSharedStorage {
    alignas(16) ElementInput A[PipelineStageNum][ThreadBlockShape::kMK];
    alignas(16) ElementInput B[PipelineStageNum][ThreadBlockShape::kKN];
};


static int constexpr EpilogueSharedStorageSize = WarpCount::kK == 1 ? 1 : WarpCount::kMN * kRowsEpiloguePerIteration * WarpShape::kN * (WarpCount::kK - 1);
struct EpilogueSharedStorage {
    alignas(16) ElementOutput D[16];
};

union SharedStorage {
  MmaSharedStorage main_loop;
  EpilogueSharedStorage epilogue;
};



static_assert(BlockLoadLine >= 8); // Swizzle优化计算后，这里要求一次读取超过8行
template <int BlockMemLoadIterNum>
__forceinline__ __device__ void LoadGmemToSmem(ElementInput* dst, const ElementInput* src, Index& index, int& blockOffset, const int& K, bool pred_guard = true) {    
    int idxSmem = index.tileSmemSaveSwizzleOffset;
    #pragma unroll
    for (int i = 0; i < BlockMemLoadIterNum; ++i) {
      asm volatile(
          "{\n"
          "  .reg .pred p;\n"
          "  setp.ne.b32 p, %0, 0;\n"
          "  @p cp.async.cg.shared.global.L2::128B [%1], [%2], %3;\n"
          "}\n" ::"r"((int)pred_guard),
          "r"((uint32_t)__cvta_generic_to_shared(&dst[idxSmem])), "l"(&src[blockOffset]), "n"(16));
        blockOffset += BlockLoadLine * K;
        idxSmem += BlockLoadLine * ThreadBlockShape::kK;
    }
    blockOffset = blockOffset - BlockLoadLine * K * BlockMemLoadIterNum + ThreadBlockShape::kK;
}



__forceinline__ __device__ void LoadSmemToRegA(ElementInput* dst, const ElementInput* src, Index& index, const int& mmaIterK) {
    int accessOffset = index.blockAWarpSwizzleOffset[mmaIterK];
    uint32_t* dstPtr = (uint32_t*)dst;
    #pragma unroll
    for (int i = 0; i < MmaIterations::kM; ++i) {
        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];" 
            : 
            "=r"(dstPtr[0]), 
            "=r"(dstPtr[1]), 
            "=r"(dstPtr[2]), 
            "=r"(dstPtr[3])
            : 
            "r"((uint32_t)__cvta_generic_to_shared(&src[accessOffset]))
        );

        dstPtr += InstructionFragNumA;
        accessOffset += InstructionShape::kM * ThreadBlockShape::kK;
    }
}

__forceinline__ __device__ void LoadSmemToRegB(ElementInput* dst, const ElementInput* src, Index& index, const int& mmaIterK) {
    int accessOffset = index.blockBWarpSwizzleOffset[mmaIterK];
    uint32_t* dstPtr = (uint32_t*)dst;
    #pragma unroll
    for (int i = 0; i < MmaIterations::kN; ++i) {
        asm volatile(
                "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];" 
                : 
                "=r"(dstPtr[0]), 
                "=r"(dstPtr[1])
                :
                "r"((uint32_t)__cvta_generic_to_shared(&src[accessOffset])) 
            );
        
        dstPtr += InstructionFragNumB;
        accessOffset += InstructionShape::kN * ThreadBlockShape::kK;
    }
}

__forceinline__ __device__ void Mma(ElementOutput* d, const ElementInput* a, const ElementInput* b,
                  const ElementOutput* c) {
    uint32_t const *A = reinterpret_cast<uint32_t const *>(a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(b);
    float const *C = reinterpret_cast<float const *>(c);
    float *D = reinterpret_cast<float *>(d);

    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));

}

__forceinline__ __device__ void WarpMma(const ElementInput* mmaFragA, const ElementInput* mmaFragB, ElementOutput* mmaFragC) {
    
    int n_serpentine = 0;
    #pragma unroll
    for (int m = 0; m < MmaIterations::kM; ++m) {
        #pragma unroll
        for (int n = 0; n < MmaIterations::kN; ++n) {
            n_serpentine = ((m % 2) ? (MmaIterations::kN - 1 - n) : n);
            Mma(&mmaFragC[(n_serpentine + m * MmaIterations::kN) * InstructionFragNumC],
              &mmaFragA[m * InstructionFragNumA],
              &mmaFragB[n_serpentine * InstructionFragNumB],
              &mmaFragC[(n_serpentine + m * MmaIterations::kN) * InstructionFragNumC]);
        }
    }
}

static_assert(ThreadBlockShape::kK == WarpShape::kK);  // Epilogue阶段没有做reduce，所以要求K维必须相等
__forceinline__ __device__ void EpilogueNoReduce(ElementOutput* dst, const ElementOutput* mmaFrag, Index& index, const int& N) {
    #pragma unroll
    for (int m = 0; m < MmaIterations::kM; ++m) {
        //int dstOffset = index.blockDOffset + m * InstructionShape::kM * N;
        int idx = 0;
        #pragma unroll
        for (int n = 0; n < MmaIterations::kN; ++n) {
            *reinterpret_cast<float2*>(&dst[index.blockDOffset]) = *reinterpret_cast<const float2*>(&mmaFrag[idx]);
            index.blockDOffset += InstructionShape::kN;
            idx += InstructionFragNumC;
        }
        index.blockDOffset -= MmaIterations::kN * InstructionShape::kN;
        index.blockDOffset += InstructionWriteRow * N;
        idx = InstructionFragWriteEleC;
        #pragma unroll
        for (int n = 0; n < MmaIterations::kN; ++n) {
            *reinterpret_cast<float2*>(&dst[index.blockDOffset]) = *reinterpret_cast<const float2*>(&mmaFrag[idx]);
            index.blockDOffset += InstructionShape::kN;
            idx += InstructionFragNumC;
        }
        index.blockDOffset -= MmaIterations::kN * InstructionShape::kN;
        index.blockDOffset += (InstructionShape::kM - InstructionWriteRow) * N;

        mmaFrag += MmaIterations::kN * InstructionFragNumC;
    }
}

// __device__ void EpilogueNoReduce(ElementOutput* dst, const ElementOutput* mmaFrag, Index& index, int N) {
//     #pragma unroll
//     for (int m = 0; m < MmaIterations::kM; ++m) {
//         int dstOffset = index.blockDOffset + m * InstructionShape::kM * N;
//         #pragma unroll
//         for (int n = 0; n < MmaIterations::kN; ++n) {
//             *reinterpret_cast<float2*>(&dst[dstOffset + n * InstructionShape::kN]) = *reinterpret_cast<const float2*>(&mmaFrag[(m * MmaIterations::kN + n) * InstructionFragNumC]);
//         }
//         dstOffset += InstructionWriteRow * N;
//         #pragma unroll
//         for (int n = 0; n < MmaIterations::kN; ++n) {
//             *reinterpret_cast<float2*>(&dst[dstOffset + n * InstructionShape::kN]) = *reinterpret_cast<const float2*>(&mmaFrag[(m * MmaIterations::kN + n) * InstructionFragNumC + InstructionFragWriteEleC]);
//         }
//     }
// }


__forceinline__ __device__ void prologue(MmaSharedStorage& mmaSharedStorage, const ElementInput* __restrict__ A, const ElementInput* __restrict__ B, 
    Index& index, const int& K) {
  #pragma unroll
  for(int i = 0; i < PipelineStageNum - 1; ++i) {
      LoadGmemToSmem<BlockMemLoadIterNumA>(mmaSharedStorage.A[i], A, index, index.blockAOffset, K);
      LoadGmemToSmem<BlockMemLoadIterNumB>(mmaSharedStorage.B[i], B, index, index.blockBOffset, K);
      asm volatile("cp.async.commit_group;\n" ::);
  }

}

__global__ void CustomGemmKernel(const ElementInput* __restrict__ A, const ElementInput* __restrict__ B, ElementOutput* __restrict__ C, int M, int N, int K, float alpha, float beta) {
    extern __shared__ int SharedStorageBase[];
    SharedStorage *shared_storage =
      reinterpret_cast<SharedStorage *>(SharedStorageBase);
  

    struct Index index(M, N, K);

    alignas(16) ElementInput mmaFragA[2][MmaIterations::kM * InstructionFragNumA];
    alignas(16) ElementInput mmaFragB[2][MmaIterations::kN * InstructionFragNumB];
    alignas(8) ElementOutput mmaFragC[MmaIterations::kMN * InstructionFragNumC] = {0.0};


    prologue(shared_storage->main_loop, A, B, index, K);


    asm volatile("cp.async.wait_group %0;\n" ::"n"(PipelineStageNum - 2));
    __syncthreads();
    

    LoadSmemToRegA(mmaFragA[0], shared_storage->main_loop.A[0], index, 0);
    LoadSmemToRegB(mmaFragB[0], shared_storage->main_loop.B[0], index, 0);


    int gemm_k_iterations = K / ThreadBlockShape::kK - PipelineStageNum;

    int smem_index = 0;

    for (; gemm_k_iterations > -PipelineStageNum; --gemm_k_iterations) {

        #pragma unroll
        for (int warp_mma_k = 0; warp_mma_k < MmaIterations::kK; ++warp_mma_k) {
            int fragIndex = (warp_mma_k + 1) % 2;
            int iter = (warp_mma_k + 1) % MmaIterations::kK;

            LoadSmemToRegA(mmaFragA[fragIndex], shared_storage->main_loop.A[smem_index], index, iter);
            LoadSmemToRegB(mmaFragB[fragIndex], shared_storage->main_loop.B[smem_index], index, iter);

            fragIndex = warp_mma_k % 2;
            WarpMma(mmaFragA[fragIndex], mmaFragB[fragIndex], mmaFragC);

            if(warp_mma_k == MmaIterations::kK - 2) {
              smem_index = (smem_index + 3) & 0x03;
              LoadGmemToSmem<BlockMemLoadIterNumA>(shared_storage->main_loop.A[smem_index], A, index, index.blockAOffset, K, gemm_k_iterations >= 0);
              LoadGmemToSmem<BlockMemLoadIterNumB>(shared_storage->main_loop.B[smem_index], B, index, index.blockBOffset, K, gemm_k_iterations >= 0);
              asm volatile("cp.async.commit_group;\n" ::);
              asm volatile("cp.async.wait_group %0;\n" ::"n"(PipelineStageNum - 2));
              
              smem_index = (smem_index + 2) & 0x03;
              __syncthreads();
            }
        }
    }
    //__syncthreads();
    EpilogueNoReduce(C, mmaFragC, index, N);
}



cudaError_t CustomGemm(ElementInput* A, ElementInput* B, ElementOutput* C, int M, int N, int K, float alpha, float beta) {

  assert(M % ThreadBlockShape::kM == 0);
  assert(N % ThreadBlockShape::kN == 0);
  dim3 grid(M / ThreadBlockShape::kM, N / ThreadBlockShape::kN, 1);
  dim3 block(ThreadBlockShape::kM / WarpShape::kM * ThreadBlockShape::kN / WarpShape::kN * WarpThreadNum, 1, 1);


  int smem_size = int(sizeof(SharedStorage));

  if (smem_size >= (48 << 10)) {
    cudaError_t result = cudaFuncSetAttribute(CustomGemmKernel,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      smem_size);
    if (result != cudaSuccess) {
      printf("cudaFuncSetAttribute failed\n");
      return result;
    }
  }

  CustomGemmKernel<<<grid, block, smem_size>>>(A, B, C, M, N, K, alpha, beta);

  return cudaGetLastError();

}

cudaError_t CutlassGemm(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta) {
  cudaError_t result;
  cutlass::Status status;

  int lda = K;
  int ldb = K;
  int ldc = N;

  using ColumnMajor = cutlass::layout::ColumnMajor;

  using RowMajor = cutlass::layout::RowMajor;

  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  using EpilogueOp = typename cutlass::gemm::device::DefaultGemmConfiguration<
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, float, float, float,
        float>::EpilogueOutputOp;

  constexpr int NumStages = 4;

  using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                  RowMajor,  // Layout of A matrix
                                                  float,        // Data-type of B matrix
                                                  ColumnMajor,  // Layout of B matrix
                                                  float,        // Data-type of C matrix
                                                  RowMajor,  // Layout of C matrix
                                                  float,
                                                  cutlass::arch::OpClassTensorOp,
                                                  cutlass::arch::Sm80,
                                                  cutlass::gemm::GemmShape<128, 128, 16>,
                                                  cutlass::gemm::GemmShape<64, 64, 16>,
                                                  cutlass::gemm::GemmShape<16, 8, 8>,
                                                  EpilogueOp,
                                                  SwizzleThreadBlock,
                                                  NumStages>;  //NumStages

  CutlassGemm::Arguments args({M , N, K},  // Gemm Problem dimensions
                              {A, lda},    // Tensor-ref for source matrix A
                              {B, ldb},    // Tensor-ref for source matrix B
                              {C, ldc},    // Tensor-ref for source matrix C
                              {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                              {alpha, beta}); // Scalars used in the Epilogue

  CutlassGemm gemm_operator;

  //status = gemm_operator(args, nullptr, stream);
  status = gemm_operator(args);

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  return cudaSuccess;

}


cudaError_t CublasGemm(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta) {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSgemm(
      handle,
      CUBLAS_OP_T, CUBLAS_OP_N,
      M, N, K,
      &alpha,
      A, M, // lda = m
      B, K, // ldb = k
      &beta,
      C, M  // ldc = m
));
  cublasDestroy(handle);
    return cudaSuccess;
}

cudaError_t ProfileGemm(int nIter, const std::function<cudaError_t(void)>& gemm_operator, int M, int N, int K) {
    cudaError_t result;
    cutlass::Status status;

    cudaEvent_t start, end;

    result = cudaEventCreate(&start);

    if (result != cudaSuccess) {
      printf("cudaEventCreate failed: %s\n", cudaGetErrorString(result));
      return result;
    }

    result = cudaEventCreate(&end);

    if (result != cudaSuccess) {
      printf("cudaEventCreate failed: %s\n", cudaGetErrorString(result));
      return result;
    }

    //warn up 
    for(int i = 0; i < 5; i++) {
        result = gemm_operator(); 
        if (result != cudaSuccess) {
            printf("Gemm failed: %s\n", cudaGetErrorString(result));
            return result;
        }
    }

    cudaEventRecord(start, 0);

    //run nIter times
    for(int i = 0; i < nIter; i++) {
        result = gemm_operator(); 
        if (result != cudaSuccess) {
            printf("Gemm failed: %s\n", cudaGetErrorString(result));
            return result;
        }
    }

    cudaEventRecord(end, 0);

    cudaEventSynchronize(end);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, end);

    double runtime = double(elapsedTime) / double(nIter);

    std::cout << "GEMM time: " << runtime << " ms. " << double(M) * K * N * 2 / 1000000000.0 << " GFLOP. " <<  double(M) * K * N * 2 / 1000000.0 / runtime << " GFLOPS." << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return cudaSuccess;
}




int main(int argc, const char *arg[]) {

  //
  // Parse the command line to obtain GEMM dimensions and scalar values.
  //

  // GEMM problem dimensions.
  int problem[3] = { 8192, 8192, 8192 };

  for (int i = 1; i < argc && i < 4; ++i) {
    std::stringstream ss(arg[i]);
    ss >> problem[i - 1];
  }

  // Scalars used for linear scaling the result of the matrix product.
  float scalars[2] = { 1, 0 };

  for (int i = 4; i < argc && i < 6; ++i) {
    std::stringstream ss(arg[i]);
    ss >> scalars[i - 4];
  }

  int nIter = 50;

  for (int i = 6; i < argc && i < 7; ++i) {
    std::stringstream ss(arg[i]);
    ss >> nIter;
  }


  cudaError_t result;
  cutlass::Status status;


  ElementInput *A;
  ElementInput *B;

  float *A2;
  float *B2;

  ElementOutput *C1;
  ElementOutput *C2;

  printf("start AllocateMatrix: A\n");

  result = AllocateMatrix(&A, problem[0], problem[2], -1);

  if (result !=  cudaSuccess) {
    return result;
  }

  printf("start AllocateMatrix: B\n");


  result = AllocateMatrix(&B, problem[2], problem[1], -1);

  if (result !=  cudaSuccess) {
    return result;
  }

printf("start AllocateMatrix: A2\n");


  result = AllocateMatrix(&A2, problem[0], problem[2], -1);

  if (result !=  cudaSuccess) {
    return result;
  }

    printf("start AllocateMatrix: B2\n");


  result = AllocateMatrix(&B2, problem[2], problem[1], -1);

  if (result !=  cudaSuccess) {
    return result;
  }

    printf("start AllocateMatrix: C1\n");


  result = AllocateMatrix(&C1, problem[0], problem[1], 101, false);

  if (result != cudaSuccess) {
    return result;
  }

    printf("start AllocateMatrix: C2\n");


  result = AllocateMatrix(&C2, problem[0], problem[1], 101, false);

  if (result != cudaSuccess) {
    return result;
  }

  cudaDeviceSynchronize();


  // printf("Matrix A: \n");
  // PrintMatrix(A, problem[0], problem[2]);

  // cudaDeviceSynchronize();

  // printf("Matrix B: \n");
  // PrintMatrix(B, problem[1], problem[2]);

  // cudaDeviceSynchronize();


  // result = TestReg(
  //   A, B, C1,
  //   problem[0],     // GEMM M dimension
  //   problem[1],     // GEMM N dimension
  //   problem[2],     // GEMM K dimension
  //   scalars[0],     // alpha
  //   scalars[1]);      // beta

  // result = CustomGemm(
  //   A, B, C1,
  //   problem[0],     // GEMM M dimension
  //   problem[1],     // GEMM N dimension
  //   problem[2],     // GEMM K dimension
  //   scalars[0],     // alpha
  //   scalars[1]);      // beta


  // result = SimpleGemm(A2, B2, C1,
  //   problem[0],     // GEMM M dimension
  //   problem[1],     // GEMM N dimension
  //   problem[2]);      // GEMM K dimension

  printf("CustomGemm: ");
  result = ProfileGemm(nIter, std::bind(&CustomGemm, 
      A, B, C1,
    problem[0],     // GEMM M dimension
    problem[1],     // GEMM N dimension
    problem[2],     // GEMM K dimension
    scalars[0],     // alpha
    scalars[1]   // beta
  ), problem[0], problem[1], problem[2]);   


  if (result == cudaSuccess) {
    std::cout << "custom gemm OK." << std::endl;
  } else {
    std::cout << "custom gemm Failed: " << result << std::endl;
  }

  cudaDeviceSynchronize();


  //  result = CutlassGemm(
  //    A2, B2, C2,
  //    problem[0],     // GEMM M dimension
  //    problem[1],     // GEMM N dimension
  //    problem[2],     // GEMM K dimension
  //    scalars[0],     // alpha
  //    scalars[1]);      // beta

  //  result = CublasGemm(
  //    A2, B2, C2,
  //    problem[0],     // GEMM M dimension
  //    problem[1],     // GEMM N dimension
  //    problem[2],     // GEMM K dimension
  //    scalars[0],     // alpha
  //    scalars[1]);      // beta


  printf("CutlassGemm: ");
  result = ProfileGemm(nIter, std::bind(&CutlassGemm, 
      A2, B2, C2,
    problem[0],     // GEMM M dimension
    problem[1],     // GEMM N dimension
    problem[2],     // GEMM K dimension
    scalars[0],     // alpha
    scalars[1]   // beta
  ), problem[0], problem[1], problem[2]);      

  // result = SimpleGemm(A2, B2, C2,
  //   problem[0],     // GEMM M dimension
  //   problem[1],     // GEMM N dimension
  //   problem[2]);      // GEMM K dimension

  if (result == cudaSuccess) {
    std::cout << "cutlass gemm OK." << std::endl;
  } else {
    std::cout << "cutlass gemm Failed: " << result << std::endl;
  }

  cudaDeviceSynchronize();


  MatrixCompare(C1, C2, problem[0], problem[1]);

  //printf("Matrix C1: \n");
  //result = PrintMatrix(C1, problem[0], problem[1]);
  //   if (result == cudaSuccess) {
  //   std::cout << "print Passed." << std::endl;
  // } else {
  //   std::cout << "print Failed: " << result  << std::endl;
  // }
  cudaDeviceSynchronize();

  // Exit.
  return result == cudaSuccess ? 0 : -1;
}
