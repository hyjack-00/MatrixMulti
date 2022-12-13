#pragma once
#include <iostream>
#include <stdint.h>

// # MATRIX SIZES
//  Square Matrix
#define N 1024
#define N6 64
#define N8 256
#define N10 1024
#define N12 4096

// # MATRIX DEFINITION
//  General Matrix

template <typename T>
struct Mat_1G {
    int width, height;
    T *data;
    Mat_1G(int h=0, int w=0) {
        data = new T[w * h];
        std::cout << "Unsupported data type." << std::endl;   
    }
    ~Mat_1G() { delete[] data; }
};
template <>
struct Mat_1G<int> {
    int width, height;
    int *data;

    Mat_1G(int h=0, int w=0) : width(w), height(h) {
        data = new int[w * h];
    }
    ~Mat_1G() { delete[] data; }

    bool operator==(const Mat_1G<int> &M) {
        if (width != M.width || height != M.height) 
            return false;
        for (int i = 0; i < width * height; i++)
            if (data[i] != M.data[i])
                return false;
        return true;
    }
};
template <>
struct Mat_1G<float> {
    int width, height;
    float *data;

    Mat_1G(int h=0, int w=0) : width(w), height(h) {
        data = new float[w * h];
    }
    ~Mat_1G() { delete[] data; }

    bool operator==(const Mat_1G<float> &M) {
        static constexpr float ERR = 1e-6;
        if (width != M.width || height != M.height) 
            return false;
        for (int i = 0; i < width * height; i++)
            if (data[i] - M.data[i] < ERR && M.data[i] - data[i] < ERR)
                return false;
        return true;
    }
};
template <>
struct Mat_1G<double> {
    int width, height;
    double *data;

    Mat_1G(int h=0, int w=0) : width(w), height(h) {
        data = new double[w * h];
    }
    ~Mat_1G() { delete[] data; }

    bool operator==(const Mat_1G<double> &M) {
        static constexpr double ERR = 1e-9;
        if (width != M.width || height != M.height) 
            return false;
        for (int i = 0; i < width * height; i++)
            if (data[i] - M.data[i] < ERR && M.data[i] - data[i] < ERR)
                return false;
        return true;
    }
};

// Square Matrix
template <typename T>
struct Mat_1D
{
    int size;
    T *data;

    Mat_1D(int sz) : size(sz)
    {
        data = new T[sz * sz];
    }
    ~Mat_1D()
    {
        delete[] data;
    }
    inline T at(int i, int j)
    {
        return data[i * size + j];
    }
    inline void change(int i, int j, T val)
    {
        data[i * size + j] = val;
    }

    // Check Equality
    bool operator==(const Mat_1D<int> &M)
    {
        if (size != M.size) return false;
        for (int i = 0; i < size * size; i++)
            if (data[i] != M.data[i])
                return false;
        return true;
    }
    bool operator==(const Mat_1D<float> &M)
    {
        static constexpr float ERR = 0.00001;
        if (size != M.size) return false;
        for (int i = 0; i < size * size; i++)
            if (data[i] - M.data[i] < ERR && M.data[i] - data[i] < ERR)
                return false;
        return true;
    }
    bool operator==(const Mat_1D<double> &M)
    {
        static constexpr float ERR = 1e-6;
        if (size != M.size) return false;
        for (int i = 0; i < size * size; i++)
            if (data[i] - M.data[i] < ERR && M.data[i] - data[i] < ERR)
                return false;
        return true;
    }
};

template <typename T>
struct Mat_2D
{
    int size;
    T **data;

    Mat_2D(int sz) : size(sz)
    {
        data = new T *[sz];
        for (int i = 0; i < sz; i++)
            data[i] = new T[sz];
    }
    ~Mat_2D()
    {
        for (int i = 0; i < size; i++)
            delete[] data[i];
        delete[] data;
    }
};

template <typename T>
struct Mat_2C
{
    // Continuous memory
    int size;
    T **data;
    T *mem;

    Mat_2C(int sz) : size(sz)
    {
        mem = new T[sz * sz];
        data = new T *[sz];
        for (int i = 0; i < sz; i++)
            data[i] = &mem[i * sz];
    }
    ~Mat_2C()
    {
        delete[] data;
        delete[] mem;
    }
};

// # BASIC MATRIX FUNC
//  Benchmark MM (Matrix Multiplication)

// Mat_1G
template <typename T>
void mm_1G_benchmark(T *A, T *B, T *C, int m, int p, int n) {
    for (int i = 0; i < m; i ++)
        for (int k = 0; k < p; k ++)
            for (int j = 0; j < n; j ++)
                C[i*n+j] += A[i*p+k] * B[k*n+j];
}

// Mat_1D
template <typename T>
void mm_1D_benchmark(T *A, T *B, T *C, int size)
{
    for (int i = 0; i < size; i++)
    {
        for (int k = 0; k < size; k++)
        {
            for (int j = 0; j < size; j++)
            {
                C[i * size + j] += A[i * size + k] * B[k * size + j];
            }
        }
    }
}

// Mat_2C
template <typename T>
void mm_2C_benchmark(T **A, T **B, T **C, int size)
{
    for (int i = 0; i < size; i++)
    {
        for (int k = 0; k < size; k++)
        {
            for (int j = 0; j < size; j++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}
