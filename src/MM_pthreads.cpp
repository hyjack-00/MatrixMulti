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
    int p = arg->p, n = arg->n;
    int ib = arg->ibegin, jb = arg->jbegin, kb = arg->kbegin;
    int ie = arg->iend, je = arg->jend, ke = arg->kend;
    int *A = arg->A, *B = arg->B, *C = arg->C;

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
}

void * pthr_G_kernel_neon_f32(void * arg_p) {

}

#else
void * pthr_G_kernel_neon_s32(void *) { return NULL; }
void * pthr_G_kernel_neon_f32(void *) { return NULL; }

#endif



//# Threads ALlocation

// 4 Threads, ChessBoard on MN
template <typename T>
void mm_G_pthread_4t_22chess(Mat_1G<T> &A, Mat_1G<T> &B, Mat_1G<T> &C, Func_t func_mm) {
    pthread_t *threads;
    threads = new pthread_t[4];

    int m = A.height, p = B.height, n = B.width;
    int m2 = m/2, n2 = n/2;
    T *a = A.data, *b = B.data, *c = C.data;

    Arg_G<T> *args[4];
    args[0] = new Arg_G<T>(a, b, c, m, p, n, 0,  0,  0, m2, n2, p);
    args[1] = new Arg_G<T>(a, b, c, m, p, n, m2, 0,  0, m,  n2, p);
    args[2] = new Arg_G<T>(a, b, c, m, p, n, 0,  n2, 0, m2, n,  p);
    args[3] = new Arg_G<T>(a, b, c, m, p, n, m2, n2, 0, m,  n,  p);
    
    for (int t = 0; t < 4; t ++)
        pthread_create(&threads[t], NULL, func_mm, args[t]);
    for (int t = 0; t < 4; t ++)
        pthread_join(threads[t], NULL);
    
    for (int t = 0; t < 4; t ++) 
        delete args[t];
    delete[] threads;
}

// 4 Threads, Split 4 on M
template <typename T>
void mm_G_pthread_4t_41split(Mat_1G<T> &A, Mat_1G<T> &B, Mat_1G<T> &C, Func_t func_mm) {
    pthread_t *threads;
    threads = new pthread_t[4];

    int m = A.height, p = B.height, n = B.width;
    int m1 = m/4, m2 = m/2, m3 = m1+m2;
    T *a = A.data, *b = B.data, *c = C.data;

    Arg_G<T> *args[4];
    args[0] = new Arg_G<T>(a, b, c, m, p, n, 0,  0,  0, m1, n,  p);
    args[1] = new Arg_G<T>(a, b, c, m, p, n, m1, 0,  0, m2, n,  p);
    args[2] = new Arg_G<T>(a, b, c, m, p, n, m2, 0,  0, m3, n,  p);
    args[3] = new Arg_G<T>(a, b, c, m, p, n, m3, 0,  0, m,  n,  p);
    
    for (int t = 0; t < 4; t ++)
        pthread_create(&threads[t], NULL, func_mm, args[t]);
    for (int t = 0; t < 4; t ++)
        pthread_join(threads[t], NULL);
    
    for (int t = 0; t < 4; t ++) 
        delete args[t];
    delete[] threads;
} 

// 测试 pthreads 非计算耗时
template <typename T>
void mm_G_pthread_fake(Mat_1G<T> &A, Mat_1G<T> &B, Mat_1G<T> &C, Func_t func_mm) {
    pthread_t *threads;
    threads = new pthread_t[1];

    int m = A.height, p = B.height, n = B.width;
    int m2 = m/2, n2 = n/2;
    T *a = A.data, *b = B.data, *c = C.data;

    Arg_G<T> *args[4];
    args[0] = new Arg_G<T>(a, b, c, m, p, n, 0,  0,  0, m, n, p);
        args[1] = new Arg_G<T>(a, b, c, m, p, n, m2, 0,  0, m,  n, p);
        args[2] = new Arg_G<T>(a, b, c, m, p, n, 0,  n2, 0, m2, n,  p);
        args[3] = new Arg_G<T>(a, b, c, m, p, n, m2, n2, 0, m,  n,  p);
    
    pthread_create(&threads[0], NULL, func_mm, args[0]);
    pthread_join(threads[0], NULL);
    
    for (int t = 0; t < 4; t ++) 
        delete args[t];
    delete[] threads;
}

// 4 Threads, Split 4 on N
template <typename T>
void mm_G_pthread_4t_14split(Mat_1G<T> &A, Mat_1G<T> &B, Mat_1G<T> &C, Func_t func_mm) {
    pthread_t *threads;
    threads = new pthread_t[4];

    int m = A.height, p = B.height, n = B.width;
    int n1 = n/4, n2 = n/2, n3 = n1+n2;
    T *a = A.data, *b = B.data, *c = C.data;

    Arg_G<T> *args[4];
    args[0] = new Arg_G<T>(a, b, c, m, p, n, 0,  0,  0,  m,  n1, p);
    args[1] = new Arg_G<T>(a, b, c, m, p, n, 0,  n1, 0,  m,  n2, p);
    args[2] = new Arg_G<T>(a, b, c, m, p, n, 0,  n2, 0,  m,  n3, p);
    args[3] = new Arg_G<T>(a, b, c, m, p, n, 0,  n3, 0,  m,  n,  p);
    
    for (int t = 0; t < 4; t ++)
        pthread_create(&threads[t], NULL, func_mm, args[t]);
    for (int t = 0; t < 4; t ++)
        pthread_join(threads[t], NULL);
    
    for (int t = 0; t < 4; t ++) 
        delete args[t];
    delete[] threads;
} 

// main + 3 = 4 threads
template <typename T>
void mm_G_pthread_1t_3t(Mat_1G<T> &A, Mat_1G<T> &B, Mat_1G<T> &C, Func_t func_mm) {
    pthread_t *threads;
    threads = new pthread_t[3];

    int m = A.height, p = B.height, n = B.width;
    int m2 = m/2, n2 = n/2;
    T *a = A.data, *b = B.data, *c = C.data;

    Arg_G<T> *args[4];
    args[0] = new Arg_G<T>(a, b, c, m, p, n, 0,  0,  0, m2, n2, p);
    args[1] = new Arg_G<T>(a, b, c, m, p, n, m2, 0,  0, m,  n2, p);
    args[2] = new Arg_G<T>(a, b, c, m, p, n, 0,  n2, 0, m2, n,  p);
    args[3] = new Arg_G<T>(a, b, c, m, p, n, m2, n2, 0, m,  n,  p);
    
    for (int t = 0; t < 3; t ++)
        pthread_create(&threads[t], NULL, func_mm, args[t]);

    func_mm(args[3]);

    for (int t = 0; t < 3; t ++)
        pthread_join(threads[t], NULL);
    
    for (int t = 0; t < 4; t ++) 
        delete args[t];
    delete[] threads;
}

// 3 threads, 31 spluit on M
template <typename T>
void mm_G_pthread_3t(Mat_1G<T> &A, Mat_1G<T> &B, Mat_1G<T> &C, Func_t func_mm) {
    pthread_t *threads;
    threads = new pthread_t[3];

    int m = A.height, p = B.height, n = B.width;
    int m1 = m/3, m2 = m1*2;
    T *a = A.data, *b = B.data, *c = C.data;

    Arg_G<T> *args[3];
    args[0] = new Arg_G<T>(a, b, c, m, p, n, 0,  0,  0, m1, n,  p);
    args[1] = new Arg_G<T>(a, b, c, m, p, n, m1, 0,  0, m2, n,  p);
    args[2] = new Arg_G<T>(a, b, c, m, p, n, m2, 0,  0, m,  n,  p);
    
    for (int t = 0; t < 3; t ++)
        pthread_create(&threads[t], NULL, func_mm, args[t]);

    for (int t = 0; t < 3; t ++)
        pthread_join(threads[t], NULL);
    
    for (int t = 0; t < 3; t ++) 
        delete args[t];
    delete[] threads;
}

// 8 threads, 2-2 chessboard on MN + 2 stage on P
template <typename T>
void mm_G_pthread_8t_2stage(Mat_1G<T> &A, Mat_1G<T> &B, Mat_1G<T> &C, Func_t func_mm) {
    pthread_t *threads;
    threads = new pthread_t[8];

    int m = A.height, p = B.height, n = B.width;
    int m2 = m/2, p2 = p/2, n2 = n/2;
    T *a = A.data, *b = B.data, *c = C.data;

    Arg_G<T> *args[8];
    args[0] = new Arg_G<T>(a, b, c, m, p, n, 0,  0,  0,  m2, n2, p2);
    args[1] = new Arg_G<T>(a, b, c, m, p, n, m2, 0,  0,  m,  n2, p2);
    args[2] = new Arg_G<T>(a, b, c, m, p, n, 0,  n2, 0,  m2, n,  p2);
    args[3] = new Arg_G<T>(a, b, c, m, p, n, m2, n2, 0,  m,  n,  p2);

    args[4] = new Arg_G<T>(a, b, c, m, p, n, 0,  0,  p2, m2, n2, p);
    args[5] = new Arg_G<T>(a, b, c, m, p, n, m2, 0,  p2, m,  n2, p);
    args[6] = new Arg_G<T>(a, b, c, m, p, n, 0,  n2, p2, m2, n,  p);
    args[7] = new Arg_G<T>(a, b, c, m, p, n, m2, n2, p2, m,  n,  p);

    for (int t = 0; t < 4; t ++)
        pthread_create(&threads[t], NULL, func_mm, args[t]);
    for (int t = 0; t < 4; t ++)
        pthread_join(threads[t], NULL);
    for (int t = 4; t < 8; t ++)
        pthread_create(&threads[t], NULL, func_mm, args[t]);
    for (int t = 4; t < 8; t ++)
        pthread_join(threads[t], NULL);
    
    for (int t = 0; t < 8; t ++) 
        delete args[t];
    delete[] threads;
}

// 8 threads, 2-2 chessboard on MN + 4 mutex
template <typename T>
void mm_G_pthread_8t_4mutex(Mat_1G<T> &A, Mat_1G<T> &B, Mat_1G<T> &C, Func_t func_mm) {
    pthread_t *threads;
    threads = new pthread_t[8];

    int m = A.height, p = B.height, n = B.width;
    int m2 = m/2, p2 = p/2, n2 = n/2;
    T *a = A.data, *b = B.data, *c = C.data;

    Arg_G<T> *args[8];
    args[0] = new Arg_G<T>(a, b, c, m, p, n, 0,  0,  0,  m2, n2, p2);
    args[1] = new Arg_G<T>(a, b, c, m, p, n, m2, 0,  0,  m,  n2, p2);
    args[2] = new Arg_G<T>(a, b, c, m, p, n, 0,  n2, 0,  m2, n,  p2);
    args[3] = new Arg_G<T>(a, b, c, m, p, n, m2, n2, 0,  m,  n,  p2);
    args[4] = new Arg_G<T>(a, b, c, m, p, n, 0,  0,  p2, m2, n2, p);
    args[5] = new Arg_G<T>(a, b, c, m, p, n, m2, 0,  p2, m,  n2, p);
    args[6] = new Arg_G<T>(a, b, c, m, p, n, 0,  n2, p2, m2, n,  p);
    args[7] = new Arg_G<T>(a, b, c, m, p, n, m2, n2, p2, m,  n,  p);

    for (int mtx = 0; mtx < 4; mtx ++)
        pthread_mutex_init(&mtx_8t[mtx], NULL);

    for (int t = 0; t < 8; t ++)
        pthread_create(&threads[t], NULL, func_mm, args[t]);
    for (int t = 0; t < 8; t ++)
        pthread_join(threads[t], NULL);
    
    for (int mtx = 0; mtx < 4; mtx ++) 
        pthread_mutex_destroy(&mtx_8t[mtx]);
    for (int t = 0; t < 8; t ++) 
        delete args[t];
    delete[] threads;
}

// 8 threads, 4-2 chessboard on MN
template <typename T>
void mm_G_pthread_8t_42chess(Mat_1G<T> &A, Mat_1G<T> &B, Mat_1G<T> &C, Func_t func_mm) {
    pthread_t *threads;
    threads = new pthread_t[8];

    int m = A.height, p = B.height, n = B.width;
    int m1 = m/4, m2 = m/2, m3 = m1+m2, n2 = n/2;
    T *a = A.data, *b = B.data, *c = C.data;

    Arg_G<T> *args[8];
    args[0] = new Arg_G<T>(a, b, c, m, p, n, 0,  0,  0,  m1, n2, p);
    args[1] = new Arg_G<T>(a, b, c, m, p, n, m1, 0,  0,  m2, n2, p);
    args[2] = new Arg_G<T>(a, b, c, m, p, n, m2, 0,  0,  m3, n2, p);
    args[3] = new Arg_G<T>(a, b, c, m, p, n, m3, 0,  0,  m,  n2, p);
    args[4] = new Arg_G<T>(a, b, c, m, p, n, 0,  n2, 0,  m1, n,  p);
    args[5] = new Arg_G<T>(a, b, c, m, p, n, m1, n2, 0,  m2, n,  p);
    args[6] = new Arg_G<T>(a, b, c, m, p, n, m2, n2, 0,  m3, n,  p);
    args[7] = new Arg_G<T>(a, b, c, m, p, n, m3, n2, 0,  m,  n,  p);

    for (int t = 0; t < 8; t ++)
        pthread_create(&threads[t], NULL, func_mm, args[t]);
    for (int t = 0; t < 8; t ++)
        pthread_join(threads[t], NULL);
    
    for (int t = 0; t < 8; t ++) 
        delete args[t];
    delete[] threads;
}

// 8 threads, 4-2 chessboard on NM
template <typename T>
void mm_G_pthread_8t_24chess(Mat_1G<T> &A, Mat_1G<T> &B, Mat_1G<T> &C, Func_t func_mm) {
    pthread_t *threads;
    threads = new pthread_t[8];

    int m = A.height, p = B.height, n = B.width;
    int n1 = n/4, n2 = n/2, n3 = n1+n2, m2 = m/2;
    T *a = A.data, *b = B.data, *c = C.data;

    Arg_G<T> *args[8];
    args[0] = new Arg_G<T>(a, b, c, m, p, n, 0,  0,  0,  m2, n1, p);
    args[1] = new Arg_G<T>(a, b, c, m, p, n, 0,  n1, 0,  m2, n2, p);
    args[2] = new Arg_G<T>(a, b, c, m, p, n, 0,  n2, 0,  m2, n3, p);
    args[3] = new Arg_G<T>(a, b, c, m, p, n, 0,  n3, 0,  m2, n,  p);
    args[4] = new Arg_G<T>(a, b, c, m, p, n, m2, 0,  0,  m,  n1, p);
    args[5] = new Arg_G<T>(a, b, c, m, p, n, m2, n1, 0,  m,  n2, p);
    args[6] = new Arg_G<T>(a, b, c, m, p, n, m2, n2, 0,  m,  n3, p);
    args[7] = new Arg_G<T>(a, b, c, m, p, n, m2, n3, 0,  m,  n,  p);

    for (int t = 0; t < 8; t ++)
        pthread_create(&threads[t], NULL, func_mm, args[t]);
    for (int t = 0; t < 8; t ++)
        pthread_join(threads[t], NULL);
    
    for (int t = 0; t < 8; t ++) 
        delete args[t];
    delete[] threads;
}

// 8 threads, 8 split on M
template <typename T>
void mm_G_pthread_8t_81split(Mat_1G<T> &A, Mat_1G<T> &B, Mat_1G<T> &C, Func_t func_mm) {
    pthread_t *threads;
    threads = new pthread_t[8];

    int m = A.height, p = B.height, n = B.width;
    int m1 = m/8, m2 = m/4, m3 = m1+m2, m4 = m/2;
    T *a = A.data, *b = B.data, *c = C.data;

    Arg_G<T> *args[8];
    args[0] = new Arg_G<T>(a, b, c, m, p, n, 0,  0,  0,  m1, n,  p);
    args[1] = new Arg_G<T>(a, b, c, m, p, n, m1, 0,  0,  m2, n,  p);
    args[2] = new Arg_G<T>(a, b, c, m, p, n, m2, 0,  0,  m3, n,  p);
    args[3] = new Arg_G<T>(a, b, c, m, p, n, m3, 0,  0,  m4, n,  p);
    args[4] = new Arg_G<T>(a, b, c, m, p, n, m4, 0,  0,  m4+m1, n,  p);
    args[5] = new Arg_G<T>(a, b, c, m, p, n, m4+m1, 0,  0,  m4+m2, n,  p);
    args[6] = new Arg_G<T>(a, b, c, m, p, n, m4+m2, 0,  0,  m4+m3, n,  p);
    args[7] = new Arg_G<T>(a, b, c, m, p, n, m4+m3, 0,  0,  m,  n,  p);
    
    for (int t = 0; t < 8; t ++)
        pthread_create(&threads[t], NULL, func_mm, args[t]);
    for (int t = 0; t < 8; t ++)
        pthread_join(threads[t], NULL);
    
    for (int t = 0; t < 8; t ++) 
        delete args[t];
    delete[] threads;
} 

// 16 threads, 4-4 chessboard on MN
template <typename T>
void mm_G_pthread_16t_44chess(Mat_1G<T> &A, Mat_1G<T> &B, Mat_1G<T> &C, Func_t func_mm) {
    pthread_t *threads;
    threads = new pthread_t[16];

    int m = A.height, p = B.height, n = B.width;
    int m1 = m/4, m2 = m/2, m3 = m1+m2;
    int n1 = n/4, n2 = n/2, n3 = n1+n2;
    T *a = A.data, *b = B.data, *c = C.data;

    Arg_G<T> *args[16];
    args[0] = new Arg_G<T>(a, b, c, m, p, n, 0,  0,  0,  m1, n1, p);
    args[1] = new Arg_G<T>(a, b, c, m, p, n, m1, 0,  0,  m2, n1, p);
    args[2] = new Arg_G<T>(a, b, c, m, p, n, m2, 0,  0,  m3, n1, p);
    args[3] = new Arg_G<T>(a, b, c, m, p, n, m3, 0,  0,  m,  n1, p);
    args[4] = new Arg_G<T>(a, b, c, m, p, n, 0,  n1, 0,  m1, n2, p);
    args[5] = new Arg_G<T>(a, b, c, m, p, n, m1, n1, 0,  m2, n2, p);
    args[6] = new Arg_G<T>(a, b, c, m, p, n, m2, n1, 0,  m3, n2, p);
    args[7] = new Arg_G<T>(a, b, c, m, p, n, m3, n1, 0,  m,  n2, p);
    args[8] = new Arg_G<T>(a, b, c, m, p, n, 0,  n2, 0,  m1, n3, p);
    args[9] = new Arg_G<T>(a, b, c, m, p, n, m1, n2, 0,  m2, n3, p);
    args[10]= new Arg_G<T>(a, b, c, m, p, n, m2, n2, 0,  m3, n3, p);
    args[11]= new Arg_G<T>(a, b, c, m, p, n, m3, n2, 0,  m,  n3, p);
    args[12]= new Arg_G<T>(a, b, c, m, p, n, 0,  n3, 0,  m1, n,  p);
    args[13]= new Arg_G<T>(a, b, c, m, p, n, m1, n3, 0,  m2, n,  p);
    args[14]= new Arg_G<T>(a, b, c, m, p, n, m2, n3, 0,  m3, n,  p);
    args[15]= new Arg_G<T>(a, b, c, m, p, n, m3, n3, 0,  m,  n,  p);

    for (int t = 0; t < 16; t ++)
        pthread_create(&threads[t], NULL, func_mm, args[t]);
    for (int t = 0; t < 16; t ++)
        pthread_join(threads[t], NULL);
    
    for (int t = 0; t < 16; t ++) 
        delete args[t];
    delete[] threads;
}

// s32
template void mm_G_pthread_4t_22chess(Mat_1G<int>, Mat_1G<int>, Mat_1G<int>, Func_t);
template void mm_G_pthread_4t_41split<int>(Mat_1G<int>, Mat_1G<int>, Mat_1G<int>, Func_t);
template void mm_G_pthread_fake<int>(Mat_1G<int>, Mat_1G<int>, Mat_1G<int>, Func_t);
template void mm_G_pthread_4t_14split<int>(Mat_1G<int>, Mat_1G<int>, Mat_1G<int>, Func_t);
template void mm_G_pthread_1t_3t<int>(Mat_1G<int>, Mat_1G<int>, Mat_1G<int>, Func_t);
template void mm_G_pthread_3t<int>(Mat_1G<int>, Mat_1G<int>, Mat_1G<int>, Func_t);

template void mm_G_pthread_8t_2stage<int>(Mat_1G<int>, Mat_1G<int>, Mat_1G<int>, Func_t);
template void mm_G_pthread_8t_4mutex<int>(Mat_1G<int>, Mat_1G<int>, Mat_1G<int>, Func_t);
template void mm_G_pthread_8t_42chess<int>(Mat_1G<int>, Mat_1G<int>, Mat_1G<int>, Func_t);
template void mm_G_pthread_8t_24chess<int>(Mat_1G<int>, Mat_1G<int>, Mat_1G<int>, Func_t);
template void mm_G_pthread_8t_81split<int>(Mat_1G<int>, Mat_1G<int>, Mat_1G<int>, Func_t);
template void mm_G_pthread_16t_44chess<int>(Mat_1G<int>, Mat_1G<int>, Mat_1G<int>, Func_t);
