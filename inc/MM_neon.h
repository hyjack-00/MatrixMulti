#pragma once
#include "matrix_def.h"

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

void mm_1D_i32_vec(int *A, int *B, int *C, int size);
void mm_1D_f32_vec();
void mm_1D_f64_vec();
