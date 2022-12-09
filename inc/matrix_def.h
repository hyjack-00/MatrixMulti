#pragma once

//# MATRIX SIZES
// Square Matrix
#define N   1024
#define N6  64
#define N8  256
#define N10 1024
#define N12 4096


//# MATRIX DEFINITION
// Square Matrix

template <typename T>
struct Mat_1D {
    int size;
    T *data;

    Mat_1D(int sz) : size(sz) { 
        data = new T [sz*sz]; 
    }
    ~Mat_1D() { 
        delete[] data; 
    }
    inline T at(int i, int j) { 
        return data[i * size + j]; 
    }
    inline void change(int i, int j, T val) {
        data[i * size + j] = val;
    }
    bool operator==(const Mat_1D &M) {
        if (size != M.size) return false;
        for (int i = 0; i < size*size; i ++) {
            if (data[i] != M.data[i]) {
                return false;
            }
        }
        return true;
    }
};

template <typename T>
struct Mat_2D {
    int size;
    T **data;

    Mat_2D(int sz) : size(sz) {
        data = new T *[sz];
        for (int i = 0; i < sz; i ++)
            data[i] = new T [sz];
    }
    ~Mat_2D() {
        for (int i = 0; i < size; i ++)
            delete[] data[i];
        delete[] data;
    }
};

template <typename T>
struct Mat_2C {
    // Continuous memory
    int size;
    T **data;
    T *mem;

    Mat_2C(int sz) : size(sz) {
        mem = new T [sz*sz];
        data = new T *[sz];
        for (int i = 0; i < sz; i ++)
            data[i] = &mem[i*sz];
    }
    ~Mat_2C() {
        delete[] data;
        delete[] mem;
    }
};

//# BASIC MATRIX FUNC
// Benchmark MM (Matrix Multiplication)
// Mat_1D 
template <typename T>
void mm_1D_benchmark(T *A, T *B, T *C, int size) {
    for (int i = 0; i < size; i ++) {
        for (int k = 0; k < size; k ++) {
            for (int j = 0; j < size; j ++) {
                C[i*size+j] += A[i*size+k] * B[k*size+j];
            }
        }
    }
}

// Mat_2C
template <typename T>
void mm_2C_benchmark(T **A, T **B, T **C, int size) {
    for (int i = 0; i < size; i ++) {
        for (int k = 0; k < size; k ++) {
            for (int j = 0; j < size; j ++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}
