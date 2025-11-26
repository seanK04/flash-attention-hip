#include "attention.hpp"

__global__ void flash_attention_kernel(const float* Q, const float* K, const float* V, float* O, int N, int d) {
    // TODO
}

void flash_attention_forward(const float* Q, const float* K, const float* V, float* O, int N, int d) {
    hipLaunchKernelGGL(flash_attention_kernel, dim3(1), dim3(64), 0, 0, Q, K, V, O, N, d);
}