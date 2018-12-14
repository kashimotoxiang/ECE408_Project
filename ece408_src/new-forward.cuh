#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_
// Sweeping various parameters to find best values (block sizes, amount of
// thread coarsening)
#define TILE_WIDTH 16
#define y4d(i3, i2, i1, i0)                                                    \
    y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]

#define x4d(i3, i2, i1, i0)                                                    \
    x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0)                                                    \
    k_shared[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

#include <mxnet/base.h>

namespace mxnet {
namespace op {

__constant__ float k_shared[15000];

__global__ void forward_kernel(float *y, const float *x, const float *k,
                               const int B, const int M, const int C,
                               const int H, const int W, const int K) {

    /*
    Modify this function to implement the forward pass described in
    Chapter 16. We have added an additional dimension to the tensors to
    support an entire mini-batch The goal here is to be correct AND fast. We
    have some nice #defs for you below to simplify indexing. Feel free to
    use them, or create your own.
    */
    int X_tile_width = TILE_WIDTH + K - 1;
    extern __shared__ float shmem[];
    float *shared_x = &shmem[0];

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int tile_iw = ceil(W_out / (TILE_WIDTH * 1.0));

    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

    int n = blockIdx.x;
    int m = blockIdx.y;

    int h0 = threadIdx.y;
    int w0 = threadIdx.x;
    int h_base = (blockIdx.z / tile_iw) * TILE_WIDTH;
    int w_base = (blockIdx.z % tile_iw) * TILE_WIDTH;
    int h = h_base + h0;
    int w = w_base + w0;

    // if (h < H_out && w < W_out) {
    //     shared_x[threadIdx.y * TILE_WIDTH + threadIdx.x] = x4d(n, c, h, w);
    // }
    // __syncthreads();

    float tmp = 0;

    // if (h < H_out && w < W_out) {
    for (int c = 0; c < C; c++) {
        for (int j = w; j < w_base + X_tile_width; j += TILE_WIDTH) {
            for (int i = h; i < h_base + X_tile_width; i += TILE_WIDTH) {
                shared_x[(i - h_base) * X_tile_width + j - w_base] =
                    x4d(n, c, i, j);
            }
        }
        __syncthreads();

        tmp += shared_x[(h0 + 0) * X_tile_width + w0 + 0] * k4d(m, c, 0, 0);
        tmp += shared_x[(h0 + 0) * X_tile_width + w0 + 1] * k4d(m, c, 0, 1);
        tmp += shared_x[(h0 + 0) * X_tile_width + w0 + 2] * k4d(m, c, 0, 2);
        tmp += shared_x[(h0 + 0) * X_tile_width + w0 + 3] * k4d(m, c, 0, 3);
        tmp += shared_x[(h0 + 0) * X_tile_width + w0 + 4] * k4d(m, c, 0, 4);
        tmp += shared_x[(h0 + 0) * X_tile_width + w0 + 5] * k4d(m, c, 0, 5);
        tmp += shared_x[(h0 + 0) * X_tile_width + w0 + 6] * k4d(m, c, 0, 6);

        tmp += shared_x[(h0 + 1) * X_tile_width + w0 + 0] * k4d(m, c, 1, 0);
        tmp += shared_x[(h0 + 1) * X_tile_width + w0 + 1] * k4d(m, c, 1, 1);
        tmp += shared_x[(h0 + 1) * X_tile_width + w0 + 2] * k4d(m, c, 1, 2);
        tmp += shared_x[(h0 + 1) * X_tile_width + w0 + 3] * k4d(m, c, 1, 3);
        tmp += shared_x[(h0 + 1) * X_tile_width + w0 + 4] * k4d(m, c, 1, 4);
        tmp += shared_x[(h0 + 1) * X_tile_width + w0 + 5] * k4d(m, c, 1, 5);
        tmp += shared_x[(h0 + 1) * X_tile_width + w0 + 6] * k4d(m, c, 1, 6);

        tmp += shared_x[(h0 + 2) * X_tile_width + w0 + 0] * k4d(m, c, 2, 0);
        tmp += shared_x[(h0 + 2) * X_tile_width + w0 + 1] * k4d(m, c, 2, 1);
        tmp += shared_x[(h0 + 2) * X_tile_width + w0 + 2] * k4d(m, c, 2, 2);
        tmp += shared_x[(h0 + 2) * X_tile_width + w0 + 3] * k4d(m, c, 2, 3);
        tmp += shared_x[(h0 + 2) * X_tile_width + w0 + 4] * k4d(m, c, 2, 4);
        tmp += shared_x[(h0 + 2) * X_tile_width + w0 + 5] * k4d(m, c, 2, 5);
        tmp += shared_x[(h0 + 2) * X_tile_width + w0 + 6] * k4d(m, c, 2, 6);

        tmp += shared_x[(h0 + 3) * X_tile_width + w0 + 0] * k4d(m, c, 3, 0);
        tmp += shared_x[(h0 + 3) * X_tile_width + w0 + 1] * k4d(m, c, 3, 1);
        tmp += shared_x[(h0 + 3) * X_tile_width + w0 + 2] * k4d(m, c, 3, 2);
        tmp += shared_x[(h0 + 3) * X_tile_width + w0 + 3] * k4d(m, c, 3, 3);
        tmp += shared_x[(h0 + 3) * X_tile_width + w0 + 4] * k4d(m, c, 3, 4);
        tmp += shared_x[(h0 + 3) * X_tile_width + w0 + 5] * k4d(m, c, 3, 5);
        tmp += shared_x[(h0 + 3) * X_tile_width + w0 + 6] * k4d(m, c, 3, 6);

        tmp += shared_x[(h0 + 4) * X_tile_width + w0 + 0] * k4d(m, c, 4, 0);
        tmp += shared_x[(h0 + 4) * X_tile_width + w0 + 1] * k4d(m, c, 4, 1);
        tmp += shared_x[(h0 + 4) * X_tile_width + w0 + 2] * k4d(m, c, 4, 2);
        tmp += shared_x[(h0 + 4) * X_tile_width + w0 + 3] * k4d(m, c, 4, 3);
        tmp += shared_x[(h0 + 4) * X_tile_width + w0 + 4] * k4d(m, c, 4, 4);
        tmp += shared_x[(h0 + 4) * X_tile_width + w0 + 5] * k4d(m, c, 4, 5);
        tmp += shared_x[(h0 + 4) * X_tile_width + w0 + 6] * k4d(m, c, 4, 6);

        tmp += shared_x[(h0 + 5) * X_tile_width + w0 + 0] * k4d(m, c, 5, 0);
        tmp += shared_x[(h0 + 5) * X_tile_width + w0 + 1] * k4d(m, c, 5, 1);
        tmp += shared_x[(h0 + 5) * X_tile_width + w0 + 2] * k4d(m, c, 5, 2);
        tmp += shared_x[(h0 + 5) * X_tile_width + w0 + 3] * k4d(m, c, 5, 3);
        tmp += shared_x[(h0 + 5) * X_tile_width + w0 + 4] * k4d(m, c, 5, 4);
        tmp += shared_x[(h0 + 5) * X_tile_width + w0 + 5] * k4d(m, c, 5, 5);
        tmp += shared_x[(h0 + 5) * X_tile_width + w0 + 6] * k4d(m, c, 5, 6);

        tmp += shared_x[(h0 + 6) * X_tile_width + w0 + 0] * k4d(m, c, 6, 0);
        tmp += shared_x[(h0 + 6) * X_tile_width + w0 + 1] * k4d(m, c, 6, 1);
        tmp += shared_x[(h0 + 6) * X_tile_width + w0 + 2] * k4d(m, c, 6, 2);
        tmp += shared_x[(h0 + 6) * X_tile_width + w0 + 3] * k4d(m, c, 6, 3);
        tmp += shared_x[(h0 + 6) * X_tile_width + w0 + 4] * k4d(m, c, 6, 4);
        tmp += shared_x[(h0 + 6) * X_tile_width + w0 + 5] * k4d(m, c, 6, 5);
        tmp += shared_x[(h0 + 6) * X_tile_width + w0 + 6] * k4d(m, c, 6, 6);

        __syncthreads();
    }
    if (n < B && m < M && h < H_out && w < W_out) {
        // y4d(b, m, h, w) = acc;
        y4d(n, m, h, w) = tmp;
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
    const int K = w.shape_[3];

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int tile_iw = ceil(W_out / (1.0 * TILE_WIDTH));
    const int tile_ih = ceil(H_out / (1.0 * TILE_WIDTH));
    const int Z = tile_iw * tile_ih;

    // Set the kernel dimensions
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(B, M, Z);

    size_t size_of_share =
        sizeof(float) * ((TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1));

    cudaMemcpyToSymbol(k_shared, w.dptr_, M * C * K * K * sizeof(float));

    // Call the kernel
    forward_kernel<<<gridDim, blockDim, size_of_share>>>(
        y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W, K);

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