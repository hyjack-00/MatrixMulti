#include "matrix_def.h"  // only suppport Mat_1G

//# pthreads
#include <pthread.h>
#include <iostream>

// mm_kernels
void * pthr_G_kernel_benchmark_s32(void * arg_p);
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

    // func_mm(args[3]);

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