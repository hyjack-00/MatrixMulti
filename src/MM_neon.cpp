#include "MM_neon.h"

#include <iostream>

void mm_1D_int32_vec(int *A, int *B, int *C, int size) {
    #ifdef __ARM_NEON
    std::cout << "NEON!!!" << std::endl;
    #else
    std::cout << "NO NEON ENV..." << std::endl;
    #endif
}
