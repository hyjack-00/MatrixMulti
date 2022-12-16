#include "MM_paral.h"

// mm_kernels
void * pthr_G_kernel_benchmark_s32(void *arg_p) {
    Arg_G<int> *arg = (Arg_G<int> *) arg_p;
    int p = arg->p, n = arg->n;
    int ib = arg->ibegin, jb = arg->jbegin, kb = arg->kbegin;
    int ie = arg->iend, je = arg->jend, ke = arg->kend;
    int *A = arg->A, *B = arg->B, *C = arg->C;

    for (int i = ib; i < ie; i ++) {
        for (int j = jb; j < je; j ++) {
            for (int k = kb; k < ke; k ++) {
                C[i*n + j] += A[i*p + k] * B[k*n + j];
            }
        }
    }
    return NULL;
}

#ifdef __ARM_NEON
void * pthr_G_kernel_neon_s32(void *) {
    return NULL;
}

#else
void * pthr_G_kernel_neon_s32(void *) { return NULL; }

#endif
