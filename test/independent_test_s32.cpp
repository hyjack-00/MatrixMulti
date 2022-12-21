#ifndef __ARM_NEON
#define __ARM_NEON  // For code highlighting
#endif

#ifdef __ARM_NEON

#include <iostream>
#include <arm_neon.h>
#include <pthread.h>
#include <chrono>
#include <stdlib.h>
#include <string.h>

using std::cout;
using std::endl;

#define Now std::chrono::system_clock::now()
#define Dur(start,end) static_cast<std::chrono::duration<double>>((end)-(start)).count()

struct Mat_1G_s32 {
    int32_t height, width;
    int32_t *data;

    Mat_1G_s32(int h=0, int w=0) : height(h), width(w) {
        data = new int[w * h];
    }
    ~Mat_1G_s32() { delete[] data; }
};

#define RAND_UB 100
#define RAND_LB -100
void rand_mat_1G_s32(Mat_1G_s32 &M, unsigned int seed) {
    srand(seed);
    int sz = M.width * M.height;
    for (int i = 0; i < sz; i ++) 
        M.data[i] = (rand() % (RAND_UB - RAND_LB)) + RAND_LB;
}

void mm_benchmark_s32(int *A, int *B, int *C, int m, int p, int n) {
    for (int i = 0; i < m; i ++) {
        for (int k = 0; k < p; k ++) {
            int Aik = A[i*p + k];
            for (int j = 0; j < n; j ++) {
                C[i*n + j] += Aik * B[k*n + j];
            }
        }
    }
}

struct Arg_s32 {
    int *A, *B, *C;
    int p, n;
    int ibegin, iend;

    Arg_s32(int *a, int *b, int *c, int p_, int n_, int ib, int ie) :
        A(a), B(b), C(c), p(p_), n(n_), ibegin(ib), iend(ie) {}
};

void * mm_kernel(void * arg_p) {
    // pthrd + neon
    Arg_s32 *arg = static_cast<Arg_s32 *>(arg_p);
    int32_t *A = arg->A, *B = arg->B, *C = arg->C;
    int p = arg->p, n = arg->n;
    int ib = arg->ibegin, ie = arg->iend;

    int32_t a, b, c;
    int32x4_t A0, A1, A2, A3;
    int32x4_t B0, B1, B2, B3;
    int32x4_t C0, C1, C2, C3;

    for (int i = ib; i < ie; i += 4) {
        for (int j = 0; j < n; j += 4) {
            C0 = vmovq_n_s32(0);
            C1 = vmovq_n_s32(0);
            C2 = vmovq_n_s32(0);
            C3 = vmovq_n_s32(0);

            for (int k = 0; k < p; k += 4) {
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

            c = i*n + j;
            vst1q_s32(C + c, C0);
            vst1q_s32(C + c + n, C1);
            vst1q_s32(C + c + n*2, C2);
            vst1q_s32(C + c + n*3, C3);
        }
    } 
    return NULL;
}

void mm_pthrd(Mat_1G_s32 &A, Mat_1G_s32 &B, Mat_1G_s32 &C) {
    pthread_t *threads;
    threads = new pthread_t[4];

    int m = A.height, p = B.height, n = B.width;
    int *a = A.data, *b = B.data, *c = C.data;

    Arg_s32 *args[4];
    int m1 = m/4, m2 = m/2, m3 = m1+m2;
    args[0] = new Arg_s32(a, b, c, p, n, 0,  m1);
    args[1] = new Arg_s32(a, b, c, p, n, m1, m2);
    args[2] = new Arg_s32(a, b, c, p, n, m2, m3);
    args[3] = new Arg_s32(a, b, c, p, n, m3, m);

    for (int t = 0; t < 4; t ++)
        pthread_create(&threads[t], NULL, mm_kernel, args[t]);
    for (int t = 0; t < 4; t ++)
        pthread_join(threads[t], NULL);

    for (int t = 0; t < 4; t ++) 
        delete args[t];
    delete[] threads;
}

int main() {
    int loop = 100, size = 1024;
    int m = size, p = size, n = size;
    cout << "IndTest: Loop-" << loop
         << ", M-" << m 
         << ", P-" << p 
         << ", N-" << n << endl;
    Mat_1G_s32 A(m, p), B(p, n), C(m, n);
    auto start = Now, end = Now;
    auto dur = Dur(start, end);

    rand_mat_1G_s32(A, 1234);
    rand_mat_1G_s32(B, 5678);
    memset(C.data, 0, sizeof(int)*m*n);

    start = Now;
    for (int l = 0; l < loop; l ++) {
        mm_pthrd(A, B, C);
        // mm_neon_s32(A.data, B.data, C.data, m, p, n);
    }
    end = Now;
    dur = Dur(start, end);
    cout << "Optimaized Time: " << dur << endl;
    cout << "Optimaized GFLOPS: " << (double)2*m*p*n*loop/dur/1e9 << endl;

    start = Now;
    for (int l = 0; l < loop/10; l ++) {
        mm_benchmark_s32(A.data, B.data, C.data, m, p, n);
    }
    end = Now;
    dur = Dur(start, end);
    cout << "BenchMark Time: " << dur*10 << endl;
    cout << "BenchMark GFLOPS: " << (double)2*m*p*n*loop/10/dur/1e9 << endl;
}

#endif 