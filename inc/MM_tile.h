#pragma once
#include "matrix_def.h"

template <typename T>
void mm_1D_tile_2pow(T *A, T *B, T *C, int size, int Ti, int Tj, int Tk) {
    int nii = size / Ti;
    int njj = size / Tj;
    int nkk = size / Tk;

    for (int ii = 0; ii < nii; ii += Ti) {
        for (int kk = 0; kk < nkk; kk += Tk) {
            for (int jj = 0; jj < njj; jj += Tj) {
                for (int i = ii; i < ii+Ti; i ++) {
                    for (int k = kk; k < kk+Tk; k ++) {
                        for (int j = jj; j < jj+Tj; j ++) {
                            C[i*size+j] += A[i*size+k] * B[k*size+j];
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void mm_1D_tile_2pow_reg(T *A, T *B, T *C, int size, int Ti, int Tj, int Tk) {
    int nii = size / Ti;
    int njj = size / Tj;
    int nkk = size / Tk;

    for (int ii = 0; ii < nii; ii += Ti) {
        for (int kk = 0; kk < nkk; kk += Tk) {
            for (int jj = 0; jj < njj; jj += Tj) {
                for (int i = ii; i < ii+Ti; i ++) {
                    for (int k = kk; k < kk+Tk; k ++) {
                        for (int j = jj; j < jj+Tj; j ++) {
                            C[i*size+j] += A[i*size+k] * B[k*size+j];
                        }
                    }
                }
            }
        }
    }
}