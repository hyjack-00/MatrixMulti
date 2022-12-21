#ifndef __ARM_NEON
#define __ARM_NEON  // For code highlighting
typedef float float32_t;
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

struct Mat_1G_f32 {
    int height, width;
    float32_t *data;

    Mat_1G_f32(int h=0, int w=0) : height(h), width(w) {
        data = new float32_t[w * h];
    }
    ~Mat_1G_f32() { delete[] data; }
};

#define RAND_UB 10
#define RAND_LB -10
void rand_mat_1G_f32(Mat_1G_f32 &M, unsigned int seed) {
    srand(seed);
    int sz = M.width * M.height;
    for (int i = 0; i < sz; i ++) 
        M.data[i] = ((double)rand() / ((double)RAND_MAX / (RAND_UB - RAND_LB))) + RAND_LB;
}

void mm_neon_f32(float32_t *A, float32_t *B, float32_t *C, int m, int p, int n) {
    int32_t a;
    int32_t b;
    int32_t c;
    float32x4_t A0;
    float32x4_t A1;
    float32x4_t A2;
    float32x4_t A3;
    float32x4_t B0;
    float32x4_t B1;
    float32x4_t B2;
    float32x4_t B3;
    float32x4_t C0;
    float32x4_t C1;
    float32x4_t C2;
    float32x4_t C3;

    for (int i = 0; i < m; i += 4) {
        for (int j = 0; j < n; j += 4) {
            C0 = vmovq_n_f32(0);
            C1 = vmovq_n_f32(0);
            C2 = vmovq_n_f32(0);
            C3 = vmovq_n_f32(0);

            for (int k = 0; k < p; k += 4) {
                a = i*p + k;
                b = k*n + j;

                B0 = vld1q_f32(B + b);
				B1 = vld1q_f32(B + b + n);
				B2 = vld1q_f32(B + b + n*2);
				B3 = vld1q_f32(B + b + n*3);

                A0 = vld1q_f32(A + a);
                C0 = vfmaq_laneq_f32(C0, B0, A0, 0);
				C0 = vfmaq_laneq_f32(C0, B1, A0, 1);
				C0 = vfmaq_laneq_f32(C0, B2, A0, 2);
				C0 = vfmaq_laneq_f32(C0, B3, A0, 3);

				A1 = vld1q_f32(A + a + p);
				C1 = vfmaq_laneq_f32(C1, B0, A1, 0);
				C1 = vfmaq_laneq_f32(C1, B1, A1, 1);
				C1 = vfmaq_laneq_f32(C1, B2, A1, 2);
				C1 = vfmaq_laneq_f32(C1, B3, A1, 3);

				A2 = vld1q_f32(A + a + p*2);
				C2 = vfmaq_laneq_f32(C2, B0, A2, 0);
				C2 = vfmaq_laneq_f32(C2, B1, A2, 1);
				C2 = vfmaq_laneq_f32(C2, B2, A2, 2);
				C2 = vfmaq_laneq_f32(C2, B3, A2, 3);

				A3 = vld1q_f32(A + a + p*3);
				C3 = vfmaq_laneq_f32(C3, B0, A3, 0);
				C3 = vfmaq_laneq_f32(C3, B1, A3, 1);
				C3 = vfmaq_laneq_f32(C3, B2, A3, 2);
				C3 = vfmaq_laneq_f32(C3, B3, A3, 3);
            }

            c = i*n + j;
            vst1q_f32(C + c, C0);
            vst1q_f32(C + c + n, C1);
            vst1q_f32(C + c + n*2, C2);
            vst1q_f32(C + c + n*3, C3);
        }
    } 
}

int main() {
    int loop = 10, size = 1024;
    int m = size, p = size, n = size;
    cout << "IndTest: Loop-" << loop
         << ", M-" << m 
         << ", P-" << p 
         << ", N-" << n << endl;
    Mat_1G_f32 A(m, p), B(p, n), C(m, n);
    auto start = Now, end = Now;
    auto dur = Dur(start, end);

    rand_mat_1G_f32(A, 65764);
    rand_mat_1G_f32(B, 56748);
    memset(C.data, 0, sizeof(float)*m*n);

    start = Now;
    for (int l = 0; l < loop; l ++) {
        mm_neon_f32(A.data, B.data, C.data, m, p, n);
    }
    end = Now;
    dur = Dur(start, end);
    cout << "Optimized Time: " << dur << endl;
    cout << "Optimized GFLOPS: " << (double)2*m*p*n*loop/dur/1e9 << endl;

}

#endif 