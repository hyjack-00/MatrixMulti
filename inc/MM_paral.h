#include "matrix_def.h"  // only suppport Mat_1G

//# pthreads
#include <pthread.h>
#include <iostream>

// mm_kernels
void * pthr_G_kernel_benchmark_s32(void * arg_p);
void * pthr_G_kernel_neon_s32(void * arg_p);
void * pthr_G_kernel_neon_f32(void * arg_p);

// Data Structures
typedef void * (*Func_mm) (void *);

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

// 4 Threads, ChessBoard on Matrix C
template <typename T>
void mm_G_pthread_4t(Mat_1G<T> &A, Mat_1G<T> &B, Mat_1G<T> &C, Func_mm func_mm) {
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

// 测试 pthreads 非计算耗时
template <typename T>
void mm_G_pthread_fake(Mat_1G<T> &A, Mat_1G<T> &B, Mat_1G<T> &C, Func_mm func_mm) {
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
