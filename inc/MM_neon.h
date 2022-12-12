#pragma once
#include "matrix_def.h"
#include <stdint.h>  // int32_t...

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

void mm_1D_s32_vec(int32_t *A, int32_t *B, int32_t *C, uint32_t size);

