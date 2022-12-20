#include "MM_openmp.h"

template <typename T>
void mm_omp_benchmark(Mat<T> &A, Mat<T> &B, Mat<T> &C, int thread_num) {
    
}

template void mm_omp_benchmark<int>(Mat<int>&, Mat<int>&, Mat<int>&, int);