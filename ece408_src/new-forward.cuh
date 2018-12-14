#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define x4d(i2, i1, i0) x[b + i2 * H * W + i1 * W + i0]
#define y4d(i2, i1, i0) y[i2 * M * H_out * W_out + i1 * W_unroll + i0]

#include <mxnet/base.h>
#define TILE_WIDTH_LARGE 24
#define TILE_WIDTH_SMALL 16
#define K 7
#define K_minus_one 6
#define K_squre 49

namespace mxnet {
namespace op {

__global__ void forward_kernel_small(float* __restrict__ y, const float* __restrict__ x, const float* __restrict__ k,
                            int M, int C, int H, int W, int H_unroll, int W_unroll) {

    /*
    Modify this function to implement the forward pass described in
    Chapter 16. We have added an additional dimension to the tensors to
    support an entire mini-batch The goal here is to be correct AND fast. We
    have some nice #defs for you below to simplify indexing. Feel free to
    use them, or create your own.
    */
    __shared__ float shared_k[TILE_WIDTH_SMALL][TILE_WIDTH_SMALL];
    __shared__ float shared_x[TILE_WIDTH_SMALL][TILE_WIDTH_SMALL];

    int b = blockIdx.z * C * H * W;

    const int H_out = H - K_minus_one;
    const int W_out = W - K_minus_one;

    int tiles = (H_unroll + TILE_WIDTH_SMALL - 1) / TILE_WIDTH_SMALL * TILE_WIDTH_SMALL;

    int col = threadIdx.x + blockIdx.x * TILE_WIDTH_SMALL;
    int row = threadIdx.y + blockIdx.y * TILE_WIDTH_SMALL;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int h_base = col / W_out;
    int w_base = col % W_out;

    float tmp = 0;

    for(int i = 0; i < tiles; i += TILE_WIDTH_SMALL) {
        int col_index = (i + tx);

        if(row < M && col_index < H_unroll) {
          shared_k[ty][tx] = k[row * H_unroll + col_index];
        } else {
          shared_k[ty][tx] = 0;
        }

        int row_index = (i + ty);

        if(col < W_unroll && row_index < H_unroll) {
            int x_index_c = row_index / K_squre;
            int x_index_s = row_index % K_squre;

            int x_index_row = x_index_s / K + h_base;
            int x_index_col = x_index_s % K + w_base;

            shared_x[ty][tx] = x4d(x_index_c, x_index_row, x_index_col);
        } else {
            shared_x[ty][tx] = 0;
        
        }

    __syncthreads();

    #pragma unroll
    for(int j = 0; j < TILE_WIDTH_SMALL; j++) {
      tmp += shared_x[j][tx] * shared_k[ty][j];
    }

    __syncthreads();

  }

  if(row < M && col < W_unroll) {
    y4d(blockIdx.z, row, col) = tmp;
  }

}

__global__ void forward_kernel_large(float* __restrict__ y, const float* __restrict__ x, const float* __restrict__ k,
                            int M, int C, int H, int W, int H_unroll, int W_unroll) {
    /*
    Modify this function to implement the forward pass described in
    Chapter 16. We have added an additional dimension to the tensors to
    support an entire mini-batch The goal here is to be correct AND fast. We
    have some nice #defs for you below to simplify indexing. Feel free to
    use them, or create your own.
    */
    __shared__ float shared_k[TILE_WIDTH_LARGE][TILE_WIDTH_LARGE];
    __shared__ float shared_x[TILE_WIDTH_LARGE][TILE_WIDTH_LARGE];

    int b = blockIdx.z * C * H * W;

    const int H_out = H - K_minus_one;
    const int W_out = W - K_minus_one;

    int tiles = (H_unroll + TILE_WIDTH_LARGE - 1) / TILE_WIDTH_LARGE * TILE_WIDTH_LARGE;

    int col = threadIdx.x + blockIdx.x * TILE_WIDTH_LARGE;
    int row = threadIdx.y + blockIdx.y * TILE_WIDTH_LARGE;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int h_base = col / W_out;
    int w_base = col % W_out;

    float tmp = 0;

    for(int i = 0; i < tiles; i += TILE_WIDTH_LARGE) {
        int col_index = (i + tx);

        if(row < M && col_index < H_unroll) {
          shared_k[ty][tx] = k[row * H_unroll + col_index];
        } else {
          shared_k[ty][tx] = 0;
        }

        int row_index = (i + ty);

        if(col < W_unroll && row_index < H_unroll) {
            int x_index_c = row_index / K_squre;
            int x_index_s = row_index % K_squre;

            int x_index_row = x_index_s / K + h_base;
            int x_index_col = x_index_s % K + w_base;

            shared_x[ty][tx] = x4d(x_index_c, x_index_row, x_index_col);
        } else {
            shared_x[ty][tx] = 0;
        
        }

    __syncthreads();

    #pragma unroll
    for(int j = 0; j < TILE_WIDTH_LARGE; j++) {
      tmp += shared_x[j][tx] * shared_k[ty][j];
    }

    __syncthreads();
  }

  if(row < M && col < W_unroll) {
    y4d(blockIdx.z, row, col) = tmp;
  }

}

/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be
   called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y,
                         const mshadow::Tensor<gpu, 4, float> &x,
                         const mshadow::Tensor<gpu, 4, float> &w) {

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    // CHECK_EQ(0, 1) << "Remove this line and replace with your
    // implementation";

    // Extract the tensor dimensions into B,M,C,H,W,K
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];

    int W_unroll = (H - K_minus_one) * (W - K_minus_one);
    int H_unroll = C * K_squre;

    if(C == 12) {
        dim3 gridDim((W_unroll + TILE_WIDTH_LARGE - 1) / TILE_WIDTH_LARGE, (M + TILE_WIDTH_LARGE - 1) / TILE_WIDTH_LARGE, B);
        dim3 blockDim(TILE_WIDTH_LARGE, TILE_WIDTH_LARGE, 1);
        forward_kernel_large<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_, M, C, H, W, H_unroll, W_unroll);
    } else {
        dim3 gridDim((W_unroll + TILE_WIDTH_SMALL - 1) / TILE_WIDTH_SMALL, (M + TILE_WIDTH_SMALL - 1) / TILE_WIDTH_SMALL, B);
        dim3 blockDim(TILE_WIDTH_SMALL, TILE_WIDTH_SMALL, 1);

        forward_kernel_small<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_, M, C, H, W, H_unroll, W_unroll);
    }   

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
}

/*
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y,
             const mshadow::Tensor<gpu, 4, DType> &x,
             const mshadow::Tensor<gpu, 4, DType> &w) {
    CHECK_EQ(0, 1)
        << "Remove this line and replace it with your implementation.";
}
} // namespace op
} // namespace mxnet

#endif
