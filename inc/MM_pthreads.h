#include "matrix_def.h"  // only suppport Mat_1G

#include <pthread.h>
#include <iostream>

// mm_kernels
void * pthr_G_kernel_benchmark_s32(void * arg_p);
void * pthr_G_kernel_benchmark_4mutex_s32(void * arg_p);
// void * pthr_G_kernel_benchmark_4mutex_w_s32(void * arg_p);  // 极其慢
void * pthr_G_kernel_neon_s32(void * arg_p);
void * pthr_G_kernel_neon_f32(void * arg_p);

// Data Structures
typedef void * (*Func_t) (void *);

template<typename T>
struct Arg_G {  // 通用的参数传递对象，之后确定最佳并行算法后可以再减少内容
    T *A, *B, *C;
    int m, p, n;
    int ibegin, jbegin, kbegin;  // [begin, end)
    int iend,   jend,   kend;

    Arg_G() : A(nullptr), B(nullptr), C(nullptr) {}
    Arg_G(T *a, T *b, T *c, int m_, int p_, int n_, int ib, int jb, int kb, int ie, int je, int ke) :
        A(a),B(b),C(c),m(m_),p(p_),n(n_),ibegin(ib),jbegin(jb),kbegin(kb),iend(ie),jend(je),kend(ke) {}
};

// Threads Allocation - Arg_G

template <typename T>  // 4 Threads, ChessBoard on MN
void mm_G_pthread_4t_22chess(Mat_1G<T> &A, Mat_1G<T> &B, Mat_1G<T> &C, Func_t func_mm);

template <typename T>  // 4 Threads, Split 4 on M
void mm_G_pthread_4t_41split(Mat_1G<T> &A, Mat_1G<T> &B, Mat_1G<T> &C, Func_t func_mm);

template <typename T>  // 测试 pthreads 非计算耗时
void mm_G_pthread_fake(Mat_1G<T> &A, Mat_1G<T> &B, Mat_1G<T> &C, Func_t func_mm);

template <typename T>  // 4 Threads, Split 4 on N
void mm_G_pthread_4t_14split(Mat_1G<T> &A, Mat_1G<T> &B, Mat_1G<T> &C, Func_t func_mm);

template <typename T>  // main + 3 = 4 threads
void mm_G_pthread_1t_3t(Mat_1G<T> &A, Mat_1G<T> &B, Mat_1G<T> &C, Func_t func_mm);

template <typename T>  // 3 threads, 31 spluit on M
void mm_G_pthread_3t(Mat_1G<T> &A, Mat_1G<T> &B, Mat_1G<T> &C, Func_t func_mm);

template <typename T>  // 8 threads, 2-2 chessboard on MN + 2 stage on P
void mm_G_pthread_8t_2stage(Mat_1G<T> &A, Mat_1G<T> &B, Mat_1G<T> &C, Func_t func_mm);

extern pthread_mutex_t mtx_8t[4];

template <typename T>  // 8 threads, 2-2 chessboard on MN + 4 mutex
void mm_G_pthread_8t_4mutex(Mat_1G<T> &A, Mat_1G<T> &B, Mat_1G<T> &C, Func_t func_mm);

template <typename T>  // 8 threads, 4-2 chessboard on MN
void mm_G_pthread_8t_42chess(Mat_1G<T> &A, Mat_1G<T> &B, Mat_1G<T> &C, Func_t func_mm);

template <typename T>  // 8 threads, 4-2 chessboard on NM
void mm_G_pthread_8t_24chess(Mat_1G<T> &A, Mat_1G<T> &B, Mat_1G<T> &C, Func_t func_mm);

template <typename T>  // 8 threads, 8 split on M
void mm_G_pthread_8t_81split(Mat_1G<T> &A, Mat_1G<T> &B, Mat_1G<T> &C, Func_t func_mm);

template <typename T>  // 16 threads, 4-4 chessboard on MN
void mm_G_pthread_16t_44chess(Mat_1G<T> &A, Mat_1G<T> &B, Mat_1G<T> &C, Func_t func_mm);