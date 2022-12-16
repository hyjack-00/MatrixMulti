#pragma once
#include "matrix_def.h"
#include <stdint.h>  // int32_t...

#ifdef __ARM_NEON
#include <arm_neon.h>
#else 
typedef float float32_t;
#endif

// Basic NEON 
void mm_1D_s32_vec(int32_t *A, int32_t *B, int32_t *C, int32_t size); 
void mm_1D_f32_vec(float32_t *A, float32_t *B, float32_t *C, int32_t size); 

// General Matrix
void mm_1G_s32_vec(int32_t *A, int32_t *B, int32_t *C,
                    int32_t m, int32_t p, int32_t n);
void mm_1G_f32_vec(float32_t *A, float32_t *B, float32_t *C,
                    int32_t m, int32_t p, int32_t n);

void mm_1G_s32_vec_ptr(int32_t *A, int32_t *B, int32_t *C,
                    int32_t m, int32_t p, int32_t n);

// NEON + Tiling
void mm_1G_f32_vec_tile_noK(float32_t *A, float32_t *B, float32_t *C,
                        int32_t m, int32_t p, int32_t n,
                        int32_t Ti, int32_t Tj);

void mm_1G_f32_vec_tile(float32_t *A, float32_t *B, float32_t *C,
                        int32_t m, int32_t p, int32_t n,
                        int32_t Ti, int32_t Tj, int32_t Tk);  // 注意这个函数里没有初始化
