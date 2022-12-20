#include "MM_openmp.h"

template <typename T>
void mm_omp_benchmark(Mat_1G<T> &A, Mat_1G<T> &B, Mat_1G<T> &C, int thread_num) {

}

template void mm_omp_benchmark<int>(Mat_1G<int>&, Mat_1G<int>&, Mat_1G<int>&, int);