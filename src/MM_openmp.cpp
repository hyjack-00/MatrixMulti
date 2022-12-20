#include "MM_openmp.h"

template <typename T>
void mm_omp_benchmark(Mat_1G<T> &a, Mat_1G<T> &b, Mat_1G<T> &c, int thr_num) {
    T *A = a.data, *B = b.data, *C = c.data;
    int m = a.height, p = a.width, n = b.width;

    omp_set_num_threads(thr_num);

    #pragma omp parallel for
    for (int i = 0; i < m; i ++) {
        for (int k = 0; k < p; k ++) {
            T Aik = A[i*p + k];
            for (int j = 0; j < n; j ++) {
                C[i*p + k] += Aik * B[k*n + j];
            }
        }
    }
}
template void mm_omp_benchmark<int>(Mat_1G<int>&, Mat_1G<int>&, Mat_1G<int>&, int);