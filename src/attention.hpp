#ifndef ATTENTION_HPP
#define ATTENTION_HPP

#include <hip/hip_runtime.h>

// Constants
constexpr int TILE_BR = 64;
constexpr int TILE_BC = 64;
constexpr int HEAD_DIM = 64;

// Thread block configuration
constexpr int THREADS_X = 16;  // columns
constexpr int THREADS_Y = 16;  // rows
constexpr int THREAD_TILE_M = TILE_BR / THREADS_Y;  // 4 rows per thread
constexpr int THREAD_TILE_N = TILE_BC / THREADS_X;  // 4 cols per thread

// Host interface function
void flash_attention_forward(const float* Q, const float* K, const float* V, float* O, int N, int d);

#endif