#include "MM_neon.h"

#define MIN(x,y) (((x)<(y))?(x):(y))
using std::cout;
using std::endl;

#ifdef __ARM_NEON

void mm_1G_s32_vec(int32_t *A, int32_t *B, int32_t *C,
                   int32_t m, int32_t p, int32_t n) {
    int32_t a;
    int32_t b;
    int32_t c;

    int32x4_t A0;
    int32x4_t A1;
    int32x4_t A2;
    int32x4_t A3;
    int32x4_t B0;
    int32x4_t B1;
    int32x4_t B2;
    int32x4_t B3;
    int32x4_t C0;
    int32x4_t C1;
    int32x4_t C2;
    int32x4_t C3;

    for (int i = 0; i < m; i += 4) {
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
}

void mm_1G_f32_vec(float32_t *A, float32_t *B, float32_t *C,
                    int32_t m, int32_t p, int32_t n) {
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

void mm_1G_s32_vec_ptr(int32_t *A, int32_t *B, int32_t *C,
                   int32_t m, int32_t p, int32_t n) {
    int32_t* Ai;
    int32_t* Bi;
    int32_t* Ci;

    int32x4_t A0;
    int32x4_t A1;
    int32x4_t A2;
    int32x4_t A3;
    int32x4_t B0;
    int32x4_t B1;
    int32x4_t B2;
    int32x4_t B3;
    int32x4_t C0;
    int32x4_t C1;
    int32x4_t C2;
    int32x4_t C3;

    for (int i = 0; i < m; i += 4) {
        for (int j = 0; j < n; j += 4) {
            C0 = vmovq_n_s32(0);
            C1 = vmovq_n_s32(0);
            C2 = vmovq_n_s32(0);
            C3 = vmovq_n_s32(0);

            for (int k = 0; k < p; k += 4) {
                Ai = A + i*p + k;
                Bi = B + k*n + j;

                B0 = vld1q_s32(Bi);
				B1 = vld1q_s32(Bi + n);
				B2 = vld1q_s32(Bi + n*2);
				B3 = vld1q_s32(Bi + n*3);

                A0 = vld1q_s32(Ai);
                C0 = vmlaq_laneq_s32(C0, B0, A0, 0);
				C0 = vmlaq_laneq_s32(C0, B1, A0, 1);
				C0 = vmlaq_laneq_s32(C0, B2, A0, 2);
				C0 = vmlaq_laneq_s32(C0, B3, A0, 3);

				A1 = vld1q_s32(Ai + p);
				C1 = vmlaq_laneq_s32(C1, B0, A1, 0);
				C1 = vmlaq_laneq_s32(C1, B1, A1, 1);
				C1 = vmlaq_laneq_s32(C1, B2, A1, 2);
				C1 = vmlaq_laneq_s32(C1, B3, A1, 3);

				A2 = vld1q_s32(Ai + p*2);
				C2 = vmlaq_laneq_s32(C2, B0, A2, 0);
				C2 = vmlaq_laneq_s32(C2, B1, A2, 1);
				C2 = vmlaq_laneq_s32(C2, B2, A2, 2);
				C2 = vmlaq_laneq_s32(C2, B3, A2, 3);

				A3 = vld1q_s32(Ai + p*3);
				C3 = vmlaq_laneq_s32(C3, B0, A3, 0);
				C3 = vmlaq_laneq_s32(C3, B1, A3, 1);
				C3 = vmlaq_laneq_s32(C3, B2, A3, 2);
				C3 = vmlaq_laneq_s32(C3, B3, A3, 3);
            }

            Ci = C + i*n + j;
            vst1q_s32(Ci, C0);
            vst1q_s32(Ci + n, C1);
            vst1q_s32(Ci + n*2, C2);
            vst1q_s32(Ci + n*3, C3);
        }
    } 
}

void mm_1G_f32_vec_tile_noK(float32_t *A, float32_t *B, float32_t *C,
                        int32_t m, int32_t p, int32_t n, 
                        int32_t Ti, int32_t Tj) {
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

    for (int ii = 0; ii < m; ii += Ti) {
        int iend = MIN(m ,ii+Ti);
     for (int jj = 0; jj < n; jj += Tj) {
        int jend = MIN(n, jj+Tj);

        for (int i = ii; i < iend; i += 4) {
            for (int j = jj; j < jend; j += 4) {
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
                }  // k

                c = i*n + j;
                vst1q_f32(C + c, C0);
                vst1q_f32(C + c + n, C1);
                vst1q_f32(C + c + n*2, C2);
                vst1q_f32(C + c + n*3, C3);
            }  // j
        }  // i

     }  // jj
    }  // ii
    
}

void mm_1G_f32_vec_tile(float32_t *A, float32_t *B, float32_t *C,
                        int32_t m, int32_t p, int32_t n,
                        int32_t Ti, int32_t Tj, int32_t Tk) {
    
}

#else
// Not NEON env

#define MM mm_1G_benchmark(A, B, C, m, p, n)
void mm_1G_s32_vec(int32_t *A, int32_t *B, int32_t *C, int32_t m, int32_t p, int32_t n) { MM; }
void mm_1G_f32_vec(float32_t *A, float32_t *B, float32_t *C, int32_t m, int32_t p, int32_t n) { MM; }
void mm_1G_s32_vec_ptr(int32_t *A, int32_t *B, int32_t *C, int32_t m, int32_t p, int32_t n) { MM; }
void mm_1G_f32_vec_tile(float32_t *A, float32_t *B, float32_t *C, int32_t m, int32_t p, int32_t n, int32_t Ti, int32_t Tj, int32_t Tk) { MM; }


#endif