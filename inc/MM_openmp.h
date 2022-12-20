#include "matrix_def.h"  // only suppport Mat_1G

#include <omp.h>
#include <iostream>

template <typename T>
void mm_omp_benchmark(Mat<T> &A, Mat<T> &B, Mat<T> &C, int thread_num);