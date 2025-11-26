#pragma once
#include <hip/hip_runtime.h>

void flash_attention_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int N,
    int d
);