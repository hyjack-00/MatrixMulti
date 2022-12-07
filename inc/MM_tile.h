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

    for (register int ii = 0; ii < nii; ii += Ti) {
        for (register int kk = 0; kk < nkk; kk += Tk) {
            for (register int jj = 0; jj < njj; jj += Tj) {
                register int iend = ii+Ti;
                for (register int i = ii; i < iend; i ++) {
                    register int kend = kk+Tk;
                    for (register int k = kk; k < kend; k ++) {
                        register int Aik = A[i*size+k];
                        register int jend = jj+Tj;
                        for (register int j = jj; j < jend; j ++) {
                            C[i*size+j] += Aik * B[k*size+j];
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void mm_1D_tile_anyT(T *A, T *B, T *C, int size, int Ti, int Tj, int Tk) {
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