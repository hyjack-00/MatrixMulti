#include "MM_pthreads.h"

#ifdef __ARM_NEON
#include <arm_neon.h>
#else 
typedef float float32_t;
#endif

//# mm_kernels

void * pthr_G_kernel_benchmark_s32(void *arg_p) {
    Arg_G<int> *arg = (Arg_G<int> *) arg_p;
    int p = arg->p, n = arg->n;
    int ib = arg->ibegin, jb = arg->jbegin, kb = arg->kbegin;
    int ie = arg->iend, je = arg->jend, ke = arg->kend;
    int *A = arg->A, *B = arg->B, *C = arg->C;

    for (int i = ib; i < ie; i ++) {
        for (int k = kb; k < ke; k ++) {
            int Aik = A[i*p + k];
            for (int j = jb; j < je; j ++) {
                C[i*n + j] += Aik * B[k*n + j];
            }
        }
    }
    // for (int i = ib; i < ie; i ++) {   // 特别慢
    //     for (int j = jb; j < je; j ++) {
    //         int c = 0;
    //         for (int k = kb; k < ke; k ++) {
    //             c += A[i*p + k] * B[k*n + j];
    //         }
    //         C[i*n + j] += c;
    //     }
    // }

    return NULL;
}

pthread_mutex_t mtx_8t[4];  // definition

void * pthr_G_kernel_benchmark_4mutex_s32(void * arg_p) {
    Arg_G<int> *arg = (Arg_G<int> *) arg_p;
    int p = arg->p, n = arg->n;
    int ib = arg->ibegin, jb = arg->jbegin, kb = arg->kbegin;
    int ie = arg->iend, je = arg->jend, ke = arg->kend;
    int *A = arg->A, *B = arg->B, *C = arg->C;
    int mtx = (ib?2:0) + (jb?1:0);  // 线程号

    pthread_mutex_lock(&mtx_8t[mtx]);

    for (int i = ib; i < ie; i ++) {
        for (int k = kb; k < ke; k ++) {
            int Aik = A[i*p + k];
            for (int j = jb; j < je; j ++) {
                C[i*n + j] += Aik * B[k*n + j];
            }
        }
    }

    pthread_mutex_unlock(&mtx_8t[mtx]);

    return NULL;
}

// void * pthr_G_kernel_benchmark_4mutex_w_s32(void * arg_p) {
//     Arg_G<int> *arg = (Arg_G<int> *) arg_p;
//     int p = arg->p, n = arg->n;
//     int ib = arg->ibegin, jb = arg->jbegin, kb = arg->kbegin;
//     int ie = arg->iend, je = arg->jend, ke = arg->kend;
//     int *A = arg->A, *B = arg->B, *C = arg->C;
//     int mtx = (ib?2:0) + (jb?1:0);  // 线程号

//     for (int i = ib; i < ie; i ++) {
//         for (int k = kb; k < ke; k ++) {
//             int Aik = A[i*p + k];
//             for (int j = jb; j < je; j ++) {
//                 pthread_mutex_lock(&mtx_8t[mtx]);
//                 C[i*n + j] += Aik * B[k*n + j];
//                 pthread_mutex_unlock(&mtx_8t[mtx]);
//             }
//         }
//     }

//     return NULL;
// }


#ifdef __ARM_NEON
void * pthr_G_kernel_neon_s32(void * arg_p) {
    Arg_G<int> *arg = (Arg_G<int> *) arg_p;
    int m = arg->m, p = arg->p, n = arg->n;
    int ib = arg->ibegin, jb = arg->jbegin, kb = arg->kbegin;
    int ie = arg->iend, je = arg->jend, ke = arg->kend;
    
    int32_t *A = arg->A, *B = arg->B, *C = arg->C;

    int32_t a, b, c;
    int32x4_t A0, A1, A2, A3;
    int32x4_t B0, B1, B2, B3;
    int32x4_t C0, C1, C2, C3;

    for (int i = ib; i < ie; i += 4) {
        for (int j = jb; j < je; j += 4) {
            c = i*n + j;
            C0 = vld1q_s32(C + c);
            C1 = vld1q_s32(C + c + n);
            C2 = vld1q_s32(C + c + n*2);
            C3 = vld1q_s32(C + c + n*3);

            for (int k = kb; k < ke; k += 4) {
                a = i*p + k;
                b = k*n + j;

                B0 = vld1q_s32(B + b);
				B1 = vld1q_s32(B + b + n);
				B2 = vld1q_s32(B + b + n*2);
				B3 = vld1q_s32(B + b + n*3);

                A0 = vld1q_s32(A + a);
                C0 = vmlaq_laneq_s32(C0, B0, A0, 0);
				C0 = vmlaq_laneq_s32(C0, B1, A0, 1);
				C0 = vmlaq_laneq_s32(C0, B2, A0, 2);
				C0 = vmlaq_laneq_s32(C0, B3, A0, 3);

				A1 = vld1q_s32(A + a + p);
				C1 = vmlaq_laneq_s32(C1, B0, A1, 0);
				C1 = vmlaq_laneq_s32(C1, B1, A1, 1);
				C1 = vmlaq_laneq_s32(C1, B2, A1, 2);
				C1 = vmlaq_laneq_s32(C1, B3, A1, 3);

				A2 = vld1q_s32(A + a + p*2);
				C2 = vmlaq_laneq_s32(C2, B0, A2, 0);
				C2 = vmlaq_laneq_s32(C2, B1, A2, 1);
				C2 = vmlaq_laneq_s32(C2, B2, A2, 2);
				C2 = vmlaq_laneq_s32(C2, B3, A2, 3);

				A3 = vld1q_s32(A + a + p*3);
				C3 = vmlaq_laneq_s32(C3, B0, A3, 0);
				C3 = vmlaq_laneq_s32(C3, B1, A3, 1);
				C3 = vmlaq_laneq_s32(C3, B2, A3, 2);
				C3 = vmlaq_laneq_s32(C3, B3, A3, 3);
            }

            vst1q_s32(C + c, C0);
            vst1q_s32(C + c + n, C1);
            vst1q_s32(C + c + n*2, C2);
            vst1q_s32(C + c + n*3, C3);
        }
    }
    return NULL;
}

void * pthr_G_kernel_neon_f32(void *) {
    return NULL;
}

#else
void * pthr_G_kernel_neon_s32(void *) { return NULL; }
void * pthr_G_kernel_neon_f32(void *) { return NULL; }

#endif

