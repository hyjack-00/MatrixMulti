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
#define MAXN 0xffffff//2^24,所以矩阵规模不要大于4096
#define __static_assert

using namespace std;

struct Mat {
    int height, width;
    int32_t *data;

    Mat(int h=0, int w=0) : height(h), width(w) {
        data = new int32_t[w * h];
    }
    ~Mat() { delete[] data; }
};


template<typename tp>
struct para {
	tp* a, * b, * c;
	//下标的下界（闭）(k-L,i-M,j-N)
	int kfrom, ifrom, jfrom;
	//下标的上界（开）
	int kto, ito, jto;
	int N, L;
	para(tp* _a, tp* _b, tp* _c, int _kfrom, int  _ifrom, int  _jfrom, int  _kto, int  _ito, int  _jto, int _N, int _L) :
		a(_a), b(_b), c(_c), kfrom(_kfrom), ifrom(_ifrom), jfrom(_jfrom), kto(_kto), ito(_ito), jto(_jto), N(_N), L(_L) {}
};

#define BLOCK 128
inline void * matmul_final_tr(void* arg) {
	int N = ((para<int32_t>*)arg)->N;
	int L = ((para<int32_t>*)arg)->L;
	for (int i = ((para<int32_t>*)arg)->ifrom; i != ((para<int32_t>*)arg)->ito; i += BLOCK) {
		for (int j = ((para<int32_t>*)arg)->jfrom; j != ((para<int32_t>*)arg)->jto; j += BLOCK) {
			for (int k = ((para<int32_t>*)arg)->kfrom; k != ((para<int32_t>*)arg)->kto; k += BLOCK) {
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

				for (int i_idx = i; i_idx != i + BLOCK; i_idx += 4) {
					for (int j_idx = j; j_idx != j + BLOCK; j_idx += 4) {
						C_idx = ((para<int>*)arg)->c + N * i_idx + j_idx;
						C0 = vld1q_s32(C_idx);
						C1 = vld1q_s32(C_idx + N);
						C2 = vld1q_s32(C_idx + N*2);
						C3 = vld1q_s32(C_idx + N*3);
						for (int k_idx = k; k_idx != k + BLOCK; k_idx += 4) {
							
							cout << i_idx << " " << j_idx << " " << k_idx << endl;

							A_idx = ((para<int32_t>*)arg)->a + L * i_idx + k_idx;
							B_idx = ((para<int32_t>*)arg)->b + j_idx + N * k_idx;

							B0 = vld1q_s32(B_idx);
							B1 = vld1q_s32(B_idx + N);
							B2 = vld1q_s32(B_idx + N*2);
							B3 = vld1q_s32(B_idx + N*3);

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

							A2 = vld1q_s32(A_idx + L*2);
							C2 = vmlaq_laneq_s32(C2, B0, A2, 0);
							C2 = vmlaq_laneq_s32(C2, B1, A2, 1);
							C2 = vmlaq_laneq_s32(C2, B2, A2, 2);
							C2 = vmlaq_laneq_s32(C2, B3, A2, 3);

							A3 = vld1q_s32(A_idx + L*3);
							C3 = vmlaq_laneq_s32(C3, B0, A3, 0);
							C3 = vmlaq_laneq_s32(C3, B1, A3, 1);
							C3 = vmlaq_laneq_s32(C3, B2, A3, 2);
							C3 = vmlaq_laneq_s32(C3, B3, A3, 3);
						}
						vst1q_s32(C_idx, C0);
						vst1q_s32(C_idx + N, C1);
						vst1q_s32(C_idx + N*2, C2);
						vst1q_s32(C_idx + N*3, C3);
					}
				}
			}
		}
	}
	return nullptr;
}

inline void matmul_final(int* a, int* b, int* c, int M, int L, int N) {
	pthread_t th[8];
	para<int32_t>* arg[8];

	arg[0] = new para<int32_t>(a, b, c, 0, 0, 0, (L >> 1), (M >> 1), (N >> 1), N, L);
	arg[1] = new para<int32_t>(a, b, c, (L >> 1), (M >> 1), 0, L, M, (N >> 1), N, L);
	arg[2] = new para<int32_t>(a, b, c, 0, 0, (N >> 1), (L >> 1), (M >> 1), N, N, L);
	arg[3] = new para<int32_t>(a, b, c, (L >> 1), (M >> 1), (N >> 1), L, M, N, N, L);

	arg[4] = new para<int32_t>(a, b, c, (L >> 1), 0, 0, L, (M >> 1), (N >> 1), N, L);
	arg[5] = new para<int32_t>(a, b, c, 0, (M >> 1), 0, (L >> 1), M, (N >> 1), N, L);
	arg[6] = new para<int32_t>(a, b, c, (L >> 1), 0, (N >> 1), L, (M >> 1), N, N, L);
	arg[7] = new para<int32_t>(a, b, c, 0, (M >> 1), (N >> 1), (L >> 1), M, N, N, L);

	for (int i = 0; i < 1; i++) {
		if (pthread_create(th + i, NULL, matmul_final_tr, arg[i])) {
			printf("Create thread error!\n");
			return;
		}
	}
	for (int i = 0; i < 1; i++) {
		pthread_join(th[i], NULL);
	}
	// for (int i = 4; i < 8; i++) {
	// 	if (pthread_create(th + i, NULL, matmul_final_tr, arg[i])) {
	// 		printf("Create thread error!\n");
	// 		return;
	// 	}
	// }
	// for (int i = 4; i < 8; i++) {
	// 	pthread_join(th[i], NULL);
	// }
	return;
}

template<typename tp>
double countTime(void f(tp*,tp*,tp*,int,int,int), tp* a, tp* b, tp* c, int m, int l, int n, int loop) {

	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < loop; ++i)
		f(a, b, c, m, l, n);
	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

	return time_span.count();
}

int main() {
	srand(time_t(NULL));

	int loop = 50;
	for (int size = 128; size <= 4096; size *= 2) {
		Mat A(size, size), B(size, size), C(size, size);
		for (int i = 0; i < size*size; i ++)
			A.data[i] = rand();
		for (int i = 0; i < size*size; i ++)
			B.data[i] = rand();

		cout << size << endl;
		cout << "final: " 
		     << countTime(matmul_final, A.data, B.data, C.data, size, size, size, loop) / loop
			 << endl;
	}

	return 0;
}