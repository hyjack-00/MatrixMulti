#include "matrix_def.h"  // only suppport Mat_1G

//# pthreads
#include <pthread.h>

// mm_kernels
void * pthr_kernel_benchmark_s32(void * arg_t);
void * pthr_kernel_neon_s32(void * arg_t);
void * pthr_kernel_neon_f32(void * arg_t);

// Data Structures
typedef void * (*Func_mm) (void *);
template<typename T>
struct Arg {
    T *A, *B, *C;
    int ibegin, jbegin, kbegin;
    int iend,   jend,   kend;

    Arg(T *a, T *b, T *c, int ib, int jb, int kb, int ie, int je, int ke) :
        A(a),B(b),C(c),ibegin(ib),jbegin(jb),kbegin(kb),iend(ie),jend(je),kend(ke) {}
};

// 4 Threads, ChessBoard on Matrix C
template <typename T>
void mm_pthread_4c(Mat_1G<T> &A, Mat_1G<T> &B, Mat_1G<T> &C, Func_mm func_mm) {
    constexpr int thread_num = 4;
    pthread_t *threads;
    threads = new pthread_t[thread_num];
    
    // pthread_create
}

