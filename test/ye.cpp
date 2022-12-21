/*这是为了向量化*/

#include <iostream>
#include <chrono>
#include <iomanip>
#include <ppl.h>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <arm_neon.h>
using namespace Concurrency;
using namespace std;

#define MIN(x,y) (x<y?x:y)
#define MAX(x,y) (x>y?x:y)
#define FOR 50
#define MAXN 0xffffff//2^24,所以矩阵规模不要大于2048
#define __static_assert

int32_t a_s[MAXN] = { 0 };
int32_t b_s[MAXN] = { 0 };
int32_t c_s[MAXN] = { 0 };
//int32_t d_s[MAXN] = { 0 };
float32_t a_f[MAXN] = { 0 };
float32_t b_f[MAXN] = { 0 };
float32_t c_f[MAXN] = { 0 };
//float32_t d_f[MAXN] = { 0 };

//这里我们不检查是否是4的倍数，所以必须要保证
// m| l_ * l| n_
int same = 4;
int N(same), M(same), L(same);


//这个是没有向量化时的写法。
template<typename tp>
inline void matmul(tp* a, tp* b, tp* c) {
	register int i, j, k, km, im;
	register tp tmp;
	for (k = 0; k != L; ++k) {
		km = k * M;
		for (i = 0; i != M; ++i) {
			im = i * M;
			tmp = *((tp*)a + im + k);
			for (j = 0; j != N; ++j) {
				*((tp*)c + im + j) += *((tp*)b + km + j) * tmp;
			}
		}
	}
}

//废案。这个是自己写的。
inline void neon_0_1(int* a, int* b, int* c) {
	register int32_t i, j, k, tmp;
	register int32_t* bkm, * cim;
	int32x4_t B, C, res;
	for (k = 0; k != L; ++k) {
		bkm = b + k * M;
		for (i = 0; i != M; ++i) {
			cim = c + i * M;
			tmp = *(a + i * M + k);
			for (j = 0; j != N; j += 4) {
				C = vld1q_s32(cim + j);
				B = vld1q_s32(bkm + j);
				res = vmlaq_n_s32(C, B, tmp);
				vst1q_s32(cim + j, res);
			}
		}
	}
}

//（int,如果最外层写
// il = L * i_idx;
// in = N * i_idx; 
// 这个方法会在规模较大时变好用）
inline void neon_s32(int32_t* A, int32_t* B, int32_t* C) {
	//C+=AB;不验证可乘
	int32_t* A_idx;
	int32_t* B_idx;
	int32_t* C_idx;

	// these are the columns of a 4x4 sub matrix of A
	int32x4_t A0;
	int32x4_t A1;
	int32x4_t A2;
	int32x4_t A3;

	// these are the columns of a 4x4 sub matrix of B
	int32x4_t B0;
	int32x4_t B1;
	int32x4_t B2;
	int32x4_t B3;

	// these are the columns of a 4x4 sub matrix of C
	int32x4_t C0;
	int32x4_t C1;
	int32x4_t C2;
	int32x4_t C3;

	for (int i_idx = 0; i_idx != M; i_idx += 4) {
		for (int j_idx = 0; j_idx != N; j_idx += 4) {
			// Zero accumulators before matrix op
			C0 = vmovq_n_s32(0);
			C1 = vmovq_n_s32(0);
			C2 = vmovq_n_s32(0);
			C3 = vmovq_n_s32(0);
			for (int k_idx = 0; k_idx != L; k_idx += 4) {
				// Compute base index to 4x4 block
				A_idx = A + L * i_idx + k_idx;
				B_idx = B + j_idx + N * k_idx;

				// Load most current A values in row 
				B0 = vld1q_s32(B_idx);
				B1 = vld1q_s32(B_idx + N);
				B2 = vld1q_s32(B_idx + (N << 1));
				B3 = vld1q_s32(B_idx + N + (N << 1));

				// Multiply accumulate in 4x1 blocks, i.e. each column in C
				A0 = vld1q_s32(A_idx);
				C0 = vmlaq_laneq_s32(C0, B0, A0, 0);
				C0 = vmlaq_laneq_s32(C0, B1, A0, 1);
				C0 = vmlaq_laneq_s32(C0, B2, A0, 2);
				C0 = vmlaq_laneq_s32(C0, B3, A0, 3);

				A1 = vld1q_s32(A_idx + L);
				C1 = vmlaq_laneq_s32(C1, B0, A1, 0);
				C1 = vmlaq_laneq_s32(C1, B1, A1, 1);
				C1 = vmlaq_laneq_s32(C1, B2, A1, 2);
				C1 = vmlaq_laneq_s32(C1, B3, A1, 3);

				A2 = vld1q_s32(A_idx + (L << 1));
				C2 = vmlaq_laneq_s32(C2, B0, A2, 0);
				C2 = vmlaq_laneq_s32(C2, B1, A2, 1);
				C2 = vmlaq_laneq_s32(C2, B2, A2, 2);
				C2 = vmlaq_laneq_s32(C2, B3, A2, 3);

				A3 = vld1q_s32(A_idx + ((L << 1) + L));
				C3 = vmlaq_laneq_s32(C3, B0, A3, 0);
				C3 = vmlaq_laneq_s32(C3, B1, A3, 1);
				C3 = vmlaq_laneq_s32(C3, B2, A3, 2);
				C3 = vmlaq_laneq_s32(C3, B3, A3, 3);
			}
			// Compute base index for stores
			C_idx = C + N * i_idx + j_idx;
			vst1q_s32(C_idx, C0);
			vst1q_s32(C_idx + N, C1);
			vst1q_s32(C_idx + (N << 1), C2);
			vst1q_s32(C_idx + ((N << 1) + N), C3);
		}
	}
}

//(float)
inline void neon_f32(float32_t* A, float32_t* B, float32_t* C) {
	/*
	* Multiply matrices A and B, store the result in C.
	* It is the user's responsibility to make sure the matrices are compatible.
	*/
	float32_t* A_idx;
	float32_t* B_idx;
	float32_t* C_idx;

	// these are the columns of a 4x4 sub matrix of A
	float32x4_t A0;
	float32x4_t A1;
	float32x4_t A2;
	float32x4_t A3;

	// these are the columns of a 4x4 sub matrix of B
	float32x4_t B0;
	float32x4_t B1;
	float32x4_t B2;
	float32x4_t B3;

	// these are the columns of a 4x4 sub matrix of C
	float32x4_t C0;
	float32x4_t C1;
	float32x4_t C2;
	float32x4_t C3;

	for (int32_t i_idx = 0; i_idx != N; i_idx += 4) {
		for (int32_t j_idx = 0; j_idx != M; j_idx += 4) {
			// Zero accumulators before matrix op
			C0 = vmovq_n_f32(0);
			C1 = vmovq_n_f32(0);
			C2 = vmovq_n_f32(0);
			C3 = vmovq_n_f32(0);
			for (int32_t k_idx = 0; k_idx != L; k_idx += 4) {
				// Compute base index to 4x4 block
				A_idx = A + L * i_idx + k_idx;
				B_idx = B + j_idx + N * k_idx;

				// Load most current A values in row 
				B0 = vld1q_f32(B_idx);
				B1 = vld1q_f32(B_idx + N);
				B2 = vld1q_f32(B_idx + (N << 1));
				B3 = vld1q_f32(B_idx + (N << 1) + N);

				// Multiply accumulate in 4x1 blocks, i.e. each column in C
				A0 = vld1q_f32(A_idx);
				C0 = vfmaq_laneq_f32(C0, B0, A0, 0);
				C0 = vfmaq_laneq_f32(C0, B1, A0, 1);
				C0 = vfmaq_laneq_f32(C0, B2, A0, 2);
				C0 = vfmaq_laneq_f32(C0, B3, A0, 3);

				A1 = vld1q_f32(A_idx + L);
				C1 = vfmaq_laneq_f32(C1, B0, A1, 0);
				C1 = vfmaq_laneq_f32(C1, B1, A1, 1);
				C1 = vfmaq_laneq_f32(C1, B2, A1, 2);
				C1 = vfmaq_laneq_f32(C1, B3, A1, 3);

				A2 = vld1q_f32(A_idx + (L << 1));
				C2 = vfmaq_laneq_f32(C2, B0, A2, 0);
				C2 = vfmaq_laneq_f32(C2, B1, A2, 1);
				C2 = vfmaq_laneq_f32(C2, B2, A2, 2);
				C2 = vfmaq_laneq_f32(C2, B3, A2, 3);

				A3 = vld1q_f32(A_idx + (L << 1) + L);
				C3 = vfmaq_laneq_f32(C3, B0, A3, 0);
				C3 = vfmaq_laneq_f32(C3, B1, A3, 1);
				C3 = vfmaq_laneq_f32(C3, B2, A3, 2);
				C3 = vfmaq_laneq_f32(C3, B3, A3, 3);
			}
			// Compute base index for stores
			C_idx = C + N * i_idx + j_idx;
			vst1q_f32(C_idx, C0);
			vst1q_f32(C_idx + N, C1);
			vst1q_f32(C_idx + (N << 1), C2);
			vst1q_f32(C_idx + (N << 1) + N, C3);
		}
	}
}

inline void neon_p_s32(int32_t* A, int32_t* B, int32_t* C) {
	//C+=AB;不验证可乘
	int32_t* A_idx;
	int32_t* B_idx;
	int32_t* C_idx;

	// these are the columns of a 4x4 sub matrix of A
	int32x4_t A0;
	int32x4_t A1;
	int32x4_t A2;
	int32x4_t A3;

	// these are the columns of a 4x4 sub matrix of B
	int32x4_t B0;
	int32x4_t B1;
	int32x4_t B2;
	int32x4_t B3;

	// these are the columns of a 4x4 sub matrix of C
	int32x4_t C0;
	int32x4_t C1;
	int32x4_t C2;
	int32x4_t C3;

	parallel_for(size_t(0), size_t(N / 4), [&](size_t i_idx) {
		for (int j_idx = 0; j_idx != M; j_idx += 4) {
			// Zero accumulators before matrix op
			C0 = vmovq_n_s32(0);
			C1 = vmovq_n_s32(0);
			C2 = vmovq_n_s32(0);
			C3 = vmovq_n_s32(0);
			for (int k_idx = 0; k_idx != L; k_idx += 4) {
				// Compute base index to 4x4 block
				A_idx = A + L * i_idx * 4 + k_idx;
				B_idx = B + j_idx + N * k_idx;

				// Load most current A values in row 
				B0 = vld1q_s32(B_idx);
				B1 = vld1q_s32(B_idx + N);
				B2 = vld1q_s32(B_idx + (N << 1));
				B3 = vld1q_s32(B_idx + N + (N << 1));

				// Multiply accumulate in 4x1 blocks, i.e. each column in C
				A0 = vld1q_s32(A_idx);
				C0 = vmlaq_laneq_s32(C0, B0, A0, 0);
				C0 = vmlaq_laneq_s32(C0, B1, A0, 1);
				C0 = vmlaq_laneq_s32(C0, B2, A0, 2);
				C0 = vmlaq_laneq_s32(C0, B3, A0, 3);

				A1 = vld1q_s32(A_idx + L);
				C1 = vmlaq_laneq_s32(C1, B0, A1, 0);
				C1 = vmlaq_laneq_s32(C1, B1, A1, 1);
				C1 = vmlaq_laneq_s32(C1, B2, A1, 2);
				C1 = vmlaq_laneq_s32(C1, B3, A1, 3);

				A2 = vld1q_s32(A_idx + (L << 1));
				C2 = vmlaq_laneq_s32(C2, B0, A2, 0);
				C2 = vmlaq_laneq_s32(C2, B1, A2, 1);
				C2 = vmlaq_laneq_s32(C2, B2, A2, 2);
				C2 = vmlaq_laneq_s32(C2, B3, A2, 3);

				A3 = vld1q_s32(A_idx + (L << 1 + L));
				C3 = vmlaq_laneq_s32(C3, B0, A3, 0);
				C3 = vmlaq_laneq_s32(C3, B1, A3, 1);
				C3 = vmlaq_laneq_s32(C3, B2, A3, 2);
				C3 = vmlaq_laneq_s32(C3, B3, A3, 3);
			}
			// Compute base index for stores
			C_idx = C + N * i_idx * 4 + j_idx;
			vst1q_s32(C_idx, C0);
			vst1q_s32(C_idx + N, C1);
			vst1q_s32(C_idx + (N << 1), C2);
			vst1q_s32(C_idx + (N << 1 + N), C3);
		}
		});
}

template<typename tp>
double countTime(void f(tp*, tp*, tp*), tp*a, tp*b, tp*c) {
	auto start = std::chrono::system_clock::now();
	auto end = std::chrono::system_clock::now();
	double dure = 0.0;

	for (int i = 0; i < FOR; i++) {
		start = std::chrono::system_clock::now();
		f(a, b, c);
		end = std::chrono::system_clock::now();
		dure += (end - start).count();
	}
	
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			c[i * M + j] = 0;
		}
	}
	return dure;
}

int main() {
	srand((int)time(0));
	//初始化两个二维数组
	for (int k = 0; k < L; ++k) {
		for (int i = 0; i < M; ++i) {
			for (int j = 0; j < N; ++j) {
				a_s[i * L + k] = rand();
				b_s[k * N + j] = rand();
				a_f[i * L + k] = rand();
				b_f[k * N + j] = rand();
			}
		}
	}

	void (*s32[4])(int32_t*, int32_t*, int32_t*) = { matmul, neon_s32, neon_p_s32 };
	std::cout << "size\t" << "\t" << "banchmark:\t" << "result_neon_1:\t" << endl;
	for (int same = 256; same <= 1024; same *= 2) {
		N = same, M = same, L = same;
		std::cout << same << "\t\t";
		for (int i = 0; i < 3; i++) {
			std::cout << countTime(s32[i], a_s, b_s, c_s) << "\t";
		}
		std::cout << endl;
	}

	void (*f32[4])(float32_t*, float32_t*, float32_t*) = { matmul, neon_f32 };
	std::cout << "size\t" << "\t" << "banchmark:\t" << "result_neon_1:\t" << endl;
	for (int same = 256; same <= 1024; same *= 2) {
		N = same, M = same, L = same;
		std::cout << same << "\t\t";
		for (int i = 0; i < 2; i++) {
			std::cout << countTime(f32[i], a_f, b_f, c_f) << "\t";
		}
		std::cout << endl;
	}
}