
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_
#define TILE_WIDTH 32
#define BLOCK_WIDTH 784

#define y4d(i3, i2, i1, i0)                                                    \
    y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]

#define x4d(i3, i2, i1, i0)                                                    \
    x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]

#define k4d(i3, i2, i1, i0)                                                    \
    k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

__constant__ float k_shared[1250];

#include <mxnet/base.h>

namespace mxnet {
namespace op {

__global__ void forward_kernel(float *y, const float *x, const float *k,
                               const int B, const int M, const int C,
                               const int H, const int W, const int K) {

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire
    mini-batch The goal here is to be correct AND fast. We have some nice #defs
    for you below to simplify indexing. Feel free to use them, or create your
    own.
    */
    __shared__ float x_shared[BLOCK_WIDTH];

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int tile_iw = ceil(W_out / (TILE_WIDTH * 1.0));

    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

    double acc = 0.0;
    int n = blockIdx.x;
    int m = blockIdx.y;
    int h = (blockIdx.z / tile_iw) * blockDim.y + threadIdx.y;
    int w = (blockIdx.z % tile_iw) * blockDim.x + threadIdx.x;
    int idx =  threadIdx.x;

    x_shared[m * TILE_WIDTH + idx] = x[n * BLOCK_WIDTH + m * TILE_WIDTH + idx];
    __syncthreads();

    if (h < H_out && w < W_out) {
        for (int c = 0; c < C; c++) {
            acc = 0.0;

            // for (int p = 0; p < K; p++) {
            //     for (int q = 0; q < K; q++) {
            //         acc + = x4d(n, c, h + p, w + q) * k4d(m, c, p, q);
            //     }
            // }

            acc += x_shared[(h +  0) * 28 + (idx + 0)] *
                   k_shared[c * 25 + 0];
            acc += x_shared[(h +  0) * 28 + (idx + 1)] *
                   k_shared[c * 25 + 1];
            acc += x_shared[(h +  0) * 28 + (idx + 2)] *
                   k_shared[c * 25 + 2];
            acc += x_shared[(h +  0) * 28 + (idx + 3)] *
                   k_shared[c * 25 + 3];
            acc += x_shared[(h +  0) * 28 + (idx + 4)] *
                   k_shared[c * 25 + 4];

            acc += x_shared[(h +  1) * 28 + (idx + 0)] *
                   k_shared[c * 25 + 5];
            acc += x_shared[(h +  1) * 28 + (idx + 1)] *
                   k_shared[c * 25 + 6];
            acc += x_shared[(h +  1) * 28 + (idx + 2)] *
                   k_shared[c * 25 + 7];
            acc += x_shared[(h +  1) * 28 + (idx + 3)] *
                   k_shared[c * 25 + 8];
            acc += x_shared[(h +  1) * 28 + (idx + 4)] *
                   k_shared[c * 25 + 9];

            acc += x_shared[(h +  2) * 28 + (idx + 0)] *
                   k_shared[c * 25 + 10];
            acc += x_shared[(h +  2) * 28 + (idx + 1)] *
                   k_shared[c * 25 + 11];
            acc += x_shared[(h +  2) * 28 + (idx + 2)] *
                   k_shared[c * 25 + 12];
            acc += x_shared[(h +  2) * 28 + (idx + 3)] *
                   k_shared[c * 25 + 13];
            acc += x_shared[(h +  2) * 28 + (idx + 4)] *
                   k_shared[c * 25 + 14];

            acc += x_shared[(h +  3) * 28 + (idx + 0)] *
                   k_shared[c * 25 + 15];
            acc += x_shared[(h +  3) * 28 + (idx + 1)] *
                   k_shared[c * 25 + 16];
            acc += x_shared[(h +  3) * 28 + (idx + 2)] *
                   k_shared[c * 25 + 17];
            acc += x_shared[(h +  3) * 28 + (idx + 3)] *
                   k_shared[c * 25 + 18];
            acc += x_shared[(h +  3) * 28 + (idx + 4)] *
                   k_shared[c * 25 + 19];

            acc += x_shared[(h +  4) * 28 + (idx + 0)] *
                   k_shared[c * 25 + 20];
            acc += x_shared[(h +  4) * 28 + (idx + 1)] *
                   k_shared[c * 25 + 21];
            acc += x_shared[(h +  4) * 28 + (idx + 2)] *
                   k_shared[c * 25 + 22];
            acc += x_shared[(h +  4) * 28 + (idx + 3)] *
                   k_shared[c * 25 + 23];
            acc += x_shared[(h +  4) * 28 + (idx + 4)] *
                   k_shared[c * 25 + 24];

            y4d(n, m, h, w) += acc;
        }
    }
}

/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so
   here we specialize with only floats.
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

    cudaMemcpyToSymbol(k_shared, w.dptr_, 1250 * sizeof(float), 0,
                       cudaMemcpyDeviceToDevice);

    // Call the kernel
    forward_kernel<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H,
                                          W, K);

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