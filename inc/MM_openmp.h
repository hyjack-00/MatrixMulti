#pragma once
#include "matrix_def.h"  // only suppport Mat_1G

#include <omp.h>
#include <iostream>

template <typename T>
void mm_omp_benchmark(Mat_1G<T> &A, Mat_1G<T> &B, Mat_1G<T> &C, int thread_num);