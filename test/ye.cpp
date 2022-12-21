/*这是为了向量化*/

#include <iostream>
#include <chrono>
#include <iomanip>
#include <pthread.h>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <arm_neon.h>
#define MIN(x,y) (x<y?x:y)
#define MAX(x,y) (x>y?x:y)
#define FOR 10
#define MAXN 0xfffff//2^20,所以矩阵规模不要大于1024
#define __static_assert

int32_t a_s[MAXN] = { 0 };
int32_t b_s[MAXN] = { 0 };
int32_t c_s[MAXN] = { 0 };
int32_t d_s[MAXN] = { 0 };
int32_t e_s[MAXN] = { 0 };
//float32_t a_f[MAXN] = { 0 };
//float32_t b_f[MAXN] = { 0 };
//float32_t c_f[MAXN] = { 0 };
//float32_t d_f[MAXN] = { 0 };

//这里我们不检查是否是4的倍数，所以必须要保证
// m| l_ * l| n_
int same = 256;
int N(same), M(same), L(same);
using namespace std;

template<typename tp>
struct para {
	tp* a, * b, * c;
	//下标的下界（闭）(k-L,i-M,j-N)
	int kfrom, ifrom, jfrom;
	//下标的上界（开）
	int kto, ito, jto;
	para(tp* _a, tp* _b, tp* _c, int _kfrom, int  _ifrom, int  _jfrom, int  _kto, int  _ito, int  _jto) :
		a(_a), b(_b), c(_c), kfrom(_kfrom), ifrom(_ifrom), jfrom(_jfrom), kto(_kto), ito(_ito), jto(_jto) {}
};

template<typename tp>
inline void* matmul_th(register void* arg) {
	register int i, j, k, km, im;
	register tp tmp;
	for (k = ((para<tp>*)arg)->kfrom; k != ((para<tp>*)arg)->kto; ++k) {
		km = k * M;
		for (i = ((para<tp>*)arg)->ifrom; i != ((para<tp>*)arg)->ito; ++i) {
			im = i * M;
			tmp = *(((para<tp>*)arg)->a + im + k);
			for (j = ((para<tp>*)arg)->jfrom; j != ((para<tp>*)arg)->jto; ++j) {
				*(((para<tp>*)arg)->c + im + j) += tmp * *(((para<tp>*)arg)->b + km + j);
			}
		}
	}
}

template<typename tp>
inline void matthr(tp* a, tp* b, tp* c) {
	pthread_t th[8];
	para<int32_t>* arg[8];

	arg[0] = new para<int32_t>(a, b, c, 0, 0, 0, (L >> 1), (M >> 1), (N >> 1));
	arg[1] = new para<int32_t>(a, b, c, (L >> 1), (M >> 1), 0, L, M, (N >> 1));
	arg[2] = new para<int32_t>(a, b, c, 0, 0, (N >> 1), (L >> 1), (M >> 1), N);
	arg[3] = new para<int32_t>(a, b, c, (L >> 1), (M >> 1), (N >> 1), L, M, N);

	arg[4] = new para<int32_t>(a, b, c, (L >> 1), 0, 0, L, (M >> 1), (N >> 1));
	arg[5] = new para<int32_t>(a, b, c, 0, (M >> 1), 0, (L >> 1), M, (N >> 1));
	arg[6] = new para<int32_t>(a, b, c, (L >> 1), 0, (N >> 1), L, (M >> 1), N);
	arg[7] = new para<int32_t>(a, b, c, 0, (M >> 1), (N >> 1), (L >> 1), M, N);


	for (int i = 0; i < 4; i++) {
		if (pthread_create(th + i, NULL, matmul_th<int32_t>, arg[i])) {
			printf("Create thread error!\n");
			return;
		}
	}
	for (int i = 0; i < 4; i++) {
		pthread_join(th[i], NULL);
	}
	for (int i = 4; i < 8; i++) {
		if (pthread_create(th + i, NULL, matmul_th<int32_t>, arg[i])) {
			printf("Create thread error!\n");
			return;
		}
	}
	for (int i = 4; i < 8; i++) {
		pthread_join(th[i], NULL);
	}
	return;
}

inline void* matmul_ne_thr(register void* arg) {
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

	register int i_idx, j_idx, k_idx;
	for (i_idx = ((para<int>*)arg)->ifrom; i_idx != ((para<int>*)arg)->ito; i_idx += 4) {
		for (j_idx = ((para<int>*)arg)->jfrom; j_idx != ((para<int>*)arg)->jto; j_idx += 4) {
			// 改的是这五行
			C_idx = ((para<int>*)arg)->c + N * i_idx + j_idx;
			C0 = vld1q_s32(C_idx);
			C1 = vld1q_s32(C_idx + N);
			C2 = vld1q_s32(C_idx + (N << 1));
			C3 = vld1q_s32(C_idx + ((N << 1) + N));
			for (k_idx = ((para<int>*)arg)->kfrom; k_idx != ((para<int>*)arg)->kto; k_idx += 4) {
				// Compute base index to 4x4 block
				A_idx = ((para<int>*)arg)->a + L * i_idx + k_idx;
				B_idx = ((para<int>*)arg)->b + j_idx + N * k_idx;

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
			vst1q_s32(C_idx, C0);
			vst1q_s32(C_idx + N, C1);
			vst1q_s32(C_idx + (N << 1), C2);
			vst1q_s32(C_idx + ((N << 1) + N), C3);
		}
	}
}

inline void matthr_ne(int* a, int* b, int* c) {
	pthread_t th[8];
	para<int32_t>* arg[8];

	arg[0] = new para<int32_t>(a, b, c, 0, 0, 0, (L >> 1), (M >> 1), (N >> 1));
	arg[1] = new para<int32_t>(a, b, c, (L >> 1), (M >> 1), 0, L, M, (N >> 1));
	arg[2] = new para<int32_t>(a, b, c, 0, 0, (N >> 1), (L >> 1), (M >> 1), N);
	arg[3] = new para<int32_t>(a, b, c, (L >> 1), (M >> 1), (N >> 1), L, M, N);

	arg[4] = new para<int32_t>(a, b, c, (L >> 1), 0, 0, L, (M >> 1), (N >> 1));
	arg[5] = new para<int32_t>(a, b, c, 0, (M >> 1), 0, (L >> 1), M, (N >> 1));
	arg[6] = new para<int32_t>(a, b, c, (L >> 1), 0, (N >> 1), L, (M >> 1), N);
	arg[7] = new para<int32_t>(a, b, c, 0, (M >> 1), (N >> 1), (L >> 1), M, N);


	for (int i = 0; i < 4; i++) {
		if (pthread_create(th + i, NULL, matmul_ne_thr, arg[i])) {
			printf("Create thread error!\n");
			return;
		}
	}
	for (int i = 0; i < 4; i++) {
		pthread_join(th[i], NULL);
	}
	for (int i = 4; i < 8; i++) {
		if (pthread_create(th + i, NULL, matmul_ne_thr, arg[i])) {
			printf("Create thread error!\n");
			return;
		}
	}
	for (int i = 4; i < 8; i++) {
		pthread_join(th[i], NULL);
	}
	return;
}

template<typename tp>
void matmul_0(tp* a, tp* b, tp* c) {
	register int i, j, k, km, im;
	register tp tmp;
	for (k = 0; k != L; ++k) {
		km = k * M;
		for (i = 0; i != M; ++i) {
			im = i * M;
			tmp = *(a + im + k);
			for (j = 0; j != N; ++j) {
				*(c + im + j) += tmp * *(b + km + j);
			}
		}
	}
}

template<typename tp>
double countTime(void f(tp*, tp*, tp*), tp* a, tp* b, tp* c) {

	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < FOR; ++i) f(a, b, c);
	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
	return time_span.count();
}

int main() {
	srand(time_t(NULL));
	//初始化两个二维数组
	for (int k = 0; k < L; ++k) {
		for (int i = 0; i < M; ++i) {
			for (int j = 0; j < N; ++j) {
				a_s[i * L + k] = i;// rand();
				b_s[k * N + j] = k;// rand();
			}
		}
	}

	for (same = 256; same < 2048; same *= 2) {
		M = N = L = same;
		cout << same << ":" << endl;
		cout << countTime(matthr, a_s, b_s, c_s) << endl;
		cout << countTime(matthr_ne, a_s, b_s, d_s) << endl;
		cout << countTime(matmul_0, a_s, b_s, e_s) << endl;

		int sign = 0;
		for (int i = 0; i < M; ++i) {
			for (int k = 0; k < N; ++k) {
				if (c_s[i * N + k] != e_s[i * N + k]) {
					sign = 1;
				}
			}
		}
		cout << sign << endl;
		sign = 0;
		for (int i = 0; i < M; ++i) {
			for (int k = 0; k < N; ++k) {
				if (d_s[i * N + k] != e_s[i * N + k]) {
					sign = 1;
				}
			}
		}
		cout << sign << endl;
	}
	
	return 0;
}