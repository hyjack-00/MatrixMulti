#include "matrix_def.h"
#include "MM_tile.h"
#include "MM_neon.h"
#include "MM_pthreads.h"

#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <chrono>

#include <queue>
#include <functional>  // greater<>

using namespace std;

// File IO
#define FILE_OUTPUT false
string ouput_file = "output/output1.txt";
#if FILE_OUTPUT == true
    #include <fstream>
    #define OS ofs
    ofstream ofs;
#else
    #define OS cout
#endif 

// Simplified time counter
/* 
    auto start = Now;
    //...
    auto end = Now;
    cout << Dur(start, end) << endl;
*/
#define Now std::chrono::system_clock::now()
#define Dur(start,end) static_cast<std::chrono::duration<double>>((end)-(start)).count()

struct Rec_tile {  // 用来记录最优分块
    int Ti, Tj, Tk;
    float time;
    Rec_tile(int i, int j, int k, float t) : Ti(i), Tj(j), Tk(k), time(t) {}
};
bool operator<(const Rec_tile &r1, const Rec_tile &r2) {
    return r1.time < r2.time;
}

// Show Matrix
template <typename T> 
void show_mat_1G(Mat_1G<T> &M) {
    int h = M.height, w = M.width;
    for (int i = 0; i < h; i ++) {
        for (int j = 0; j < w; j ++)
            OS << setw(6) << fixed << setprecision(1) << M.data[i*w + j] << " ";
        OS << endl;
    }
    OS << endl;
}

// Random init
#define RAND_SEED1 20221124
#define RAND_SEED2 20221123
#define RAND_UB 10  // [LB, UB)
#define RAND_LB 0
void rand_mat_1D_s32(Mat_1D<int> &M, unsigned int seed);
void rand_mat_2C_s32(Mat_2C<int> &M, unsigned int seed);
void rand_mat_1G_s32(Mat_1G<int> &M, unsigned int seed);
void rand_mat_1G_f32(Mat_1G<float> &M, unsigned int seed);
void rand_mat_1G_f64(Mat_1G<double> &M, unsigned int seed);

// Unit test -----------------------------------------------------------

void test_pthrd_neon();
void test_pthrd();
void test_neon_tile();
void test_neon_f32();
void test_neon_s32();
void test_cal_correct();
void test_tile_reg(double &reg, double &no_reg);
void test_tile();
void test_reg_restrict();
void test_mat_access_speed();

int main() {
    cout << "Test begin." << endl;
    
    #if FILE_OUTPUT == true
        cout << "Output File: " << ouput_file << endl;
        ofs.open(ouput_file, ios::out);
    #endif
    
    // double r = 0, nr = 0;
    for (int i = 0; i < 10; i ++) {
        cout << "Test No." << i 
             << " =============================" << endl;
        // test_mat_access_speed();
        // test_reg_restrict();
        // test_tile();
        // test_tile_reg(r, nr);
        // test_cal_correct();
        // test_neon_s32();
        // test_neon_f32();
        // test_neon_tile();
        // test_pthrd();
        test_pthrd_neon();

    }
    cout << "Test end." << endl;

    #if FILE_OUTPUT == true
        ofs.close();
    #endif
}

// Test implementation -----------------------------------------------------------

void test_pthrd_neon() {
    int loop = 10, size = 1024;
    int m = size, p = size, n = size;
    OS << "Pthreads+Neon test: Loop-" << loop;
    OS << ", M-" << m << ", P-" << p << ", N-" << n << endl;
    Mat_1G<int> A(m, p), B(p, n), C(m, n), Ans(m, n);
    rand_mat_1G_s32(A, 1234);
    rand_mat_1G_s32(B, 5678);

    memset(Ans.data, 0, sizeof(int)*m*n);
    mm_1G_benchmark(A.data, B.data, Ans.data, m, p, n);
    memset(C.data, 0, sizeof(int)*m*n);
    mm_G_pthread_4t_22chess<int>(A, B, C, pthr_G_kernel_neon_s32);
    if (C == Ans) OS << "Correct" << endl;
    else          OS << "Wrong" << endl;

    auto start = Now, end = Now;

    start = Now;
    for (int l = 0; l < loop; l ++) 
        mm_G_pthread_fake<int>(A, B, C, pthr_G_kernel_neon_s32);
    end = Now;
    OS << "parallel-1 (benchmark): " << Dur(start, end) << endl;

    start = Now;
    for (int l = 0; l < loop; l ++) 
        mm_G_pthread_4t_22chess<int>(A, B, C, pthr_G_kernel_neon_s32);
    end = Now;
    OS << "parallel-4-22chess: " << Dur(start, end) << endl;

    start = Now;
    for (int l = 0; l < loop; l ++) 
        mm_G_pthread_4t_41split<int>(A, B, C, pthr_G_kernel_neon_s32);
    end = Now;
    OS << "parallel-4-41split: " << Dur(start, end) << endl;

    start = Now;
    for (int l = 0; l < loop; l ++) 
        mm_G_pthread_4t_14split<int>(A, B, C, pthr_G_kernel_neon_s32);
    end = Now;
    OS << "parallel-4-14split: " << Dur(start, end) << endl;

    start = Now;
    for (int l = 0; l < loop; l ++) 
        mm_G_pthread_8t_2stage<int>(A, B, C, pthr_G_kernel_neon_s32);
    end = Now;
    OS << "parallel-8-2stage: " << Dur(start, end) << endl;

    start = Now;
    for (int l = 0; l < loop; l ++) 
        mm_G_pthread_8t_42chess<int>(A, B, C, pthr_G_kernel_neon_s32);
    end = Now;
    OS << "parallel-8-42chess: " << Dur(start, end) << endl;

    start = Now;
    for (int l = 0; l < loop; l ++) 
        mm_G_pthread_8t_24chess<int>(A, B, C, pthr_G_kernel_neon_s32);
    end = Now;
    OS << "parallel-8-24chess: " << Dur(start, end) << endl;

    start = Now;
    for (int l = 0; l < loop; l ++) 
        mm_G_pthread_8t_81split<int>(A, B, C, pthr_G_kernel_neon_s32);
    end = Now;
    OS << "parallel-8-81split: " << Dur(start, end) << endl;
}


void test_pthrd() {
    int loop = 10, size = 1024;
    int m = size, p = size, n = size;
    OS << "Pthreads test: Loop-" << loop;
    OS << ", M-" << m << ", P-" << p << ", N-" << n << endl;
    Mat_1G<int> A(m, p), B(p, n), C(m, n), Ans(m, n);
    rand_mat_1G_s32(A, 1234);
    rand_mat_1G_s32(B, 5678);

    memset(Ans.data, 0, sizeof(int)*m*n);
    mm_1G_benchmark(A.data, B.data, Ans.data, m, p, n);
    memset(C.data, 0, sizeof(int)*m*n);
    mm_G_pthread_8t_4mutex<int>(A, B, C, pthr_G_kernel_benchmark_4mutex_s32);
    if (C == Ans) OS << "Correct" << endl;
    else          OS << "Wrong" << endl;

    // int b = 0, e = 32;  // [b,e]
    // cin >> b >> e;
    // for (int i = b; i <= e; i ++)
    //     cout << Ans.data[i] << ' ';
    // cout << endl;
    // for (int i = b; i <= e; i ++)
    //     cout << C.data[i] << ' ';
    // cout << endl;

    auto start = Now, end = Now;
    // start = Now;
    // for (int l = 0; l < loop; l ++) 
    //     mm_1G_benchmark(A.data, B.data, C.data, m, p, n);
    // end = Now;
    // OS << "benchmark: " << Dur(start, end) << endl;

    start = Now;
    for (int l = 0; l < loop; l ++) 
        mm_G_pthread_fake<int>(A, B, C, pthr_G_kernel_benchmark_s32);
    end = Now;
    OS << "parallel-1 (benchmark): " << Dur(start, end) << endl;

    start = Now;
    for (int l = 0; l < loop; l ++) 
        mm_G_pthread_4t_22chess<int>(A, B, C, pthr_G_kernel_benchmark_s32);
    end = Now;
    OS << "parallel-4-22chess: " << Dur(start, end) << endl;

    start = Now;
    for (int l = 0; l < loop; l ++) 
        mm_G_pthread_1t_3t<int>(A, B, C, pthr_G_kernel_benchmark_s32);
    end = Now;
    OS << "parallel-4=main+3: " << Dur(start, end) << endl;

    start = Now;
    for (int l = 0; l < loop; l ++) 
        mm_G_pthread_3t<int>(A, B, C, pthr_G_kernel_benchmark_s32);
    end = Now;
    OS << "parallel-3-31split: " << Dur(start, end) << endl;

    start = Now;
    for (int l = 0; l < loop; l ++) 
        mm_G_pthread_4t_41split<int>(A, B, C, pthr_G_kernel_benchmark_s32);
    end = Now;
    OS << "parallel-4-41split: " << Dur(start, end) << endl;

    start = Now;
    for (int l = 0; l < loop; l ++) 
        mm_G_pthread_4t_14split<int>(A, B, C, pthr_G_kernel_benchmark_s32);
    end = Now;
    OS << "parallel-4-14split: " << Dur(start, end) << endl;

    start = Now;
    for (int l = 0; l < loop; l ++) 
        mm_G_pthread_8t_2stage<int>(A, B, C, pthr_G_kernel_benchmark_s32);
    end = Now;
    OS << "parallel-8-2stage: " << Dur(start, end) << endl;

    start = Now;
    for (int l = 0; l < loop; l ++) 
        mm_G_pthread_8t_4mutex<int>(A, B, C, pthr_G_kernel_benchmark_4mutex_s32);
    end = Now;
    OS << "parallel-8-4mutex: " << Dur(start, end) << endl;

    // start = Now;
    // for (int l = 0; l < loop; l ++) 
    //     mm_G_pthread_8t_4mutex<int>(A, B, C, pthr_G_kernel_benchmark_4mutex_w_s32);
    // end = Now;
    // OS << "parallel-8-4mutex-w: " << Dur(start, end) << endl;

    start = Now;
    for (int l = 0; l < loop; l ++) 
        mm_G_pthread_8t_42chess<int>(A, B, C, pthr_G_kernel_benchmark_s32);
    end = Now;
    OS << "parallel-8-42chess: " << Dur(start, end) << endl;

    start = Now;
    for (int l = 0; l < loop; l ++) 
        mm_G_pthread_8t_24chess<int>(A, B, C, pthr_G_kernel_benchmark_s32);
    end = Now;
    OS << "parallel-8-24chess: " << Dur(start, end) << endl;

    start = Now;
    for (int l = 0; l < loop; l ++) 
        mm_G_pthread_8t_81split<int>(A, B, C, pthr_G_kernel_benchmark_s32);
    end = Now;
    OS << "parallel-8-81split: " << Dur(start, end) << endl;

    start = Now;
    for (int l = 0; l < loop; l ++) 
        mm_G_pthread_16t_44chess<int>(A, B, C, pthr_G_kernel_benchmark_s32);
    end = Now;
    OS << "parallel-16-44chess: " << Dur(start, end) << endl;
}

void test_neon_tile() {
    int loop = 100, size = 512;
    int m = size, p = size, n = size;
    int Ti_start = 32, Tj_start = 32, Tk_start = 32;
    int Ti_end = size, Tj_end = size, Tk_end = size;
    int Ti_step = 32, Tj_step = 32, Tk_step = 32;

    OS << "Neon + Tile test f32: Loop-" << loop;
    OS << ", M-" << m << ", P-" << p << ", N-" << n << endl;
    Mat_1G<float> A(m, p), B(p, n), C(m, n), D(m, n);
    rand_mat_1G_f32(A, RAND_SEED1);
    rand_mat_1G_f32(B, RAND_SEED2);
    memset(C.data, 0, sizeof(int)*m*n);

    // Correctness
    mm_1G_f32_vec_tile(A.data, B.data, C.data, m, p, n, 4, 16, 16);
    mm_1G_benchmark(A.data, B.data, D.data, m, p, n);
    if (C == D) OS << "Correct" << endl;
    else { OS << "Wrong!!" << endl; return; }
    
    priority_queue<Rec_tile> q;
    for (int x = 1; x <= 40; x ++) q.push(Rec_tile(0, 0, 0, 10000));  // 选取时间最少的前40

    OS << "  Ti   Tj   Tk   Time" << endl;
    for (int Tk = Tk_start; Tk <= Tk_end; Tk += Tk_step) {
        for (int Tj = Tj_start; Tj <= Tj_end; Tj += Tj_step) {
            // k 分块
            for (int Ti = Ti_start; Ti <= Ti_end; Ti += Ti_step) {
                OS << setw(4) << Ti << " " << setw(4) << Tj << " " << setw(4) << Tk << "   ";
                auto start = Now;
                for (int l = 0; l < loop; l ++) {
                    mm_1G_f32_vec_tile(A.data, B.data, C.data, m, p, n, Ti, Tj, Tk);
                }
                auto end = Now;
                double dur = Dur(start, end);
                OS << setprecision(12) << dur;
                if (dur < q.top().time) {  // 进入前40
                    q.pop();
                    q.push(Rec_tile(Ti, Tj, Tk, dur));
                    // OS << " recorded";
                }
                OS << endl;
            }
            
            // k 不分块
                // OS << setw(4) << Ti << " " << setw(4) << Tj << "    0   ";
                // auto start = Now;
                // for (int l = 0; l < loop; l ++) {
                //     mm_1G_f32_vec_tile_noK(A.data, B.data, C.data, m, p, n, Ti, Tj);
                // }
                // auto end = Now;
                // double dur = Dur(start, end);
                // OS << setprecision(12) << dur;
                // if (dur < q.top().time) {  // 进入前40
                //     q.pop();
                //     q.push(Rec_tile(Ti, Tj, 0, dur));
                //     OS << " recorded";
                // }
                // OS << endl;
        }
    }
    auto start0 = Now;
    for (int l = 0; l < loop; l ++) {
        mm_1G_f32_vec(A.data, B.data, C.data, m, p, n);
    }
    auto end0 = Now;
    OS << "vec benchmark: " << Dur(start0, end0) << endl;

    // 展示前30
    OS << endl << "====================== Speedest 40 ======================" << endl;
    OS << "  Ti   Tj   Tk   Time" << endl;
    while (!q.empty()) {
        Rec_tile r = q.top();
        q.pop();
        OS << setw(4) << r.Ti << " " << setw(4) << r.Tj << " " << setw(4) << r.Tk << "   ";
        OS << setprecision(12) << r.time << endl;
    }
}

void test_neon_f32() {
    int loop = 100, m = 1024, p = 512, n = 16;
    OS << "Neon test fp32: Loop-" << loop;
    OS << ", M-" << m << ", P-" << p << ", N-" << n << endl;
    Mat_1G<float> A(m, p), B(p, n), C(m, n), Ans(m, n);
    rand_mat_1G_f32(A, RAND_SEED1);
    rand_mat_1G_f32(B, RAND_SEED2);
    // show_mat_1G(A);
    // show_mat_1G(B);

    mm_1G_benchmark(A.data, B.data, Ans.data, m, p, n);
    // show_mat_1G(Ans);
    mm_1G_f32_vec_tile(A.data, B.data, C.data, m, p, n, m, n, p);
    // show_mat_1G(C);

    if (C == Ans) OS << "Correct" << endl;
    else          OS << "Wrong!!" << endl;
}

void test_neon_s32() {
    int loop = 100, m = 1024, p = 512, n = 512;
    OS << "Neon test: Loop-" << loop;
    OS << ", M-" << m << ", P-" << p << ", N-" << n << endl;

    Mat_1G<int> A(m, p), B(p, n), Vec(m, n), VecPtr(m, n), Ans(m, n);
    rand_mat_1G_s32(A, RAND_SEED1);
    rand_mat_1G_s32(B, RAND_SEED2);

    // Correcetness check
    mm_1G_benchmark(A.data, B.data, Ans.data, m, p, n);
    mm_1G_s32_vec(A.data, B.data, Vec.data, m, p, n);
    (Vec    == Ans) ? OS << "Correct " : OS << "Wrong!! ";
    mm_1G_s32_vec_ptr(A.data, B.data, VecPtr.data, m, p, n);
    (VecPtr == Ans) ? OS << "Correct " : OS << "Wrong!! ";
    OS << endl;

    // auto start = Now;
    // for (int i = 0; i < loop; i ++) {
    //     mm_1G_s32_vec(A.data, B.data, Vec.data, m, p, n);
    // }
    // auto end = Now;
    // OS << Dur(start, end) << endl;
    
    /// 换 ptr 无区别
    // start = Now;
    // for (int i = 0; i < loop; i ++) {
    //     mm_1G_s32_vec_ptr(A.data, B.data, Vec.data, m, p, n);
    // } 
    // end = Now;
    // OS << Dur(start, end) << endl;
    
    // start = Now;
    // for (int i = 0; i < loop; i ++) {
    //     mm_1G_benchmark(A.data, B.data, Vec.data, m, p, n);
    // }
    // end = Now;
    // OS << Dur(start, end) << endl;
}

void test_cal_correct() {
    int loop = 10, size = 1024, Tsize = 64;
    OS << "Precise tiling test: Loop-" << loop << ", Size-" << size << endl;
    Mat_1D<int> A(size), B(size), C(size), D(size), E(size);
    rand_mat_1D_s32(A, RAND_SEED1);
    rand_mat_1D_s32(B, RAND_SEED2);

    memset(E.data, 0, sizeof(int)*size*size);
    mm_1D_benchmark(A.data, B.data, E.data, size);

    for (int i = 0; i < loop; i ++) {
        memset(C.data, 0, sizeof(int)*size*size);
        memset(D.data, 0, sizeof(int)*size*size);
        mm_1D_tile_2pow(A.data, B.data, C.data, size, Tsize, Tsize, Tsize);
        mm_1D_benchmark(A.data, B.data, D.data, size);
        
        if (C == D) OS << "checked: Correct";
        if (C == E) OS << ", Consistent";

        OS << endl;
    }
}

void test_tile_reg(double &reg, double &no_reg) {
    // 结果：几乎无区别
    cout << "Tiling & register test:" << endl;
    int loop = 1000, size = 4096, Tsize = 64;
    Mat_1D<int> A(size), B(size), C(size);
    rand_mat_1D_s32(A, RAND_SEED1);
    rand_mat_1D_s32(B, RAND_SEED2);

    auto start = Now;
    for (int i = 0; i < loop; i ++) {
        memset(C.data, 0, sizeof(int)*size*size);
        mm_1D_tile_2pow_reg(A.data, B.data, C.data, size, Tsize, Tsize, Tsize);
    }
    auto end = Now;
    reg += Dur(start, end);
    cout << "Tile       : " << reg << endl;

    start = Now;
    for (int i = 0; i < loop; i ++) {
        memset(C.data, 0, sizeof(int)*size*size);
        mm_1D_tile_2pow(A.data, B.data, C.data, size, Tsize, Tsize, Tsize);
    }
    end = Now;
    no_reg += Dur(start, end);
    cout << "Tile & Reg : " << no_reg << endl;
}

void test_tile() {
    int loop = 10, size = 1024;
    OS << "Tiling test: Loop-" << loop << ", Size-" << size << endl;
    Mat_1D<int> A(size), B(size), C(size);
    rand_mat_1D_s32(A, RAND_SEED1);
    rand_mat_1D_s32(B, RAND_SEED2);

    priority_queue<Rec_tile> q;
    for (int x = 1; x <= 20; x ++) q.push(Rec_tile(0, 0, 0, 10000));  // 选取时间最少的前20

    OS << "  Ti   Tj   Tk   Time" << endl;
    for (int Ti = 512; Ti >= 32; Ti /= 2) {
        for (int Tj = 1024; Tj >= 256; Tj /= 2) {
            for (int Tk = 8; Tk >= 1; Tk /= 2) {
                OS << setw(4) << Ti << " " << setw(4) << Tj << " " << setw(4) << Tk << "   ";
                auto start = Now;
                for (int l = 0; l < loop; l ++) {
                    // memset(C.data, 0, sizeof(int)*size*size);
                    mm_1D_tile_2pow(A.data, B.data, C.data, size, Ti, Tj, Tk);
                }
                auto end = Now;
                double dur = Dur(start, end);
                OS << dur;
                if (dur < q.top().time) {  // 进入前20
                    q.pop();
                    q.push(Rec_tile(Ti, Tj, Tk, dur));
                    OS << " recorded";
                }
                OS << endl;
            }
        }
    }
    auto start = Now;
    mm_1D_benchmark(A.data, B.data, C.data, size);
    auto end = Now;
    OS << "Benchmark: " << Dur(start, end) << endl;

    // 展示前20
    OS << endl << "========== Speedest 20 ==========" << endl;
    OS << "  Ti   Tj   Tk   Time" << endl;
    while (!q.empty()) {
        Rec_tile r = q.top();
        q.pop();
        OS << setw(4) << r.Ti << " " << setw(4) << r.Tj << " " << setw(4) << r.Tk << "   ";
        OS << r.time << endl;
    }
}

void test_mat_access_speed() {
    int loop = 100, size = 4096;
    Mat_1D<int> A(size);
    Mat_2C<int> B(size);
    cout << "Access speed test: " << endl;

    // simple RW
    auto start1 = Now;
    for (int l = 0; l < loop; l ++) {
        for (int i = 0; i < size; i ++) {
            for (int j = 0; j < size; j ++) {
                A.change(i, j, 11 + A.at(i, j));
            }
        }
    }
    auto end1 = Now;
    cout << Dur(start1, end1) << endl;

    auto start2 = Now;
    int *pa = A.data;
    for (int l = 0; l < loop; l ++) {
        for (int i = 0; i < size; i ++) {
            for (int j = 0; j < size; j ++) {
                pa[i*size+j] += 11;
            }
        }
    }
    auto end2 = Now;
    cout << Dur(start2, end2) << endl;

    auto start3 = Now;
    __restrict_arr int **pb = B.data;
    for (int l = 0; l < loop; l ++) {
        for (int i = 0; i < size; i ++) {
            for (int j = 0; j < size; j ++) {
                pb[i][j] = pb[i][j] + 11;
            }
        }
    }
    auto end3 = Now;
    cout << Dur(start3, end3) << endl;
    cout << "Access speed test finish." << endl;
}

void test_reg_restrict() {
    int size = 1024, loop = 10;
    cout << "register/restrict test:" << endl;

    Mat_1D<int> A1(size), B1(size), C1(size);
    rand_mat_1D_s32(A1, RAND_SEED1);
    rand_mat_1D_s32(B1, RAND_SEED2);
    auto start = Now;
    for (int i = 0; i < loop; i ++) {
        // mm_1D_benchmark_reg(A1.data, B1.data, C1.data, size);
    }
    auto end = Now;
    cout << "Mat_1D: " << Dur(start, end) << endl;

    Mat_2C<int> A2(size), B2(size), C2(size);
    rand_mat_2C_s32(A2, RAND_SEED1);
    rand_mat_2C_s32(B2, RAND_SEED2);
    start = Now;
    for (int i = 0; i < loop; i ++) {
        mm_2C_benchmark(A2.data, B2.data, C2.data, size);
    }
    end = Now;
    cout << "Mat_2C: " << Dur(start, end) << endl;

    cout << "register/restrict test finish." << endl;
}

void rand_mat_1D_s32(Mat_1D<int> &M, unsigned int seed) {
    srand(seed);
    int sz = M.size;
    for (int i = 0; i < sz*sz; i ++) 
        M.data[i] = (rand() % (RAND_UB - RAND_LB)) + RAND_LB;
}
void rand_mat_2C_s32(Mat_2C<int> &M, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < M.size; i ++)
        for (int j = 0; j < M.size; j ++)
            M.data[i][j] = (rand() % (RAND_UB - RAND_LB)) + RAND_LB;
}
void rand_mat_1G_s32(Mat_1G<int> &M, unsigned int seed) {
    srand(seed);
    int sz = M.width * M.height;
    for (int i = 0; i < sz; i ++) 
        M.data[i] = (rand() % (RAND_UB - RAND_LB)) + RAND_LB;
}
void rand_mat_1G_f32(Mat_1G<float> &M, unsigned int seed) {
    srand(seed);
    int sz = M.width * M.height;
    for (int i = 0; i < sz; i ++) 
        M.data[i] = ((float)rand() / ((float)RAND_MAX / (RAND_UB - RAND_LB))) + RAND_LB;
}
void rand_mat_1G_f64(Mat_1G<double> &M, unsigned int seed) {
    srand(seed);
    int sz = M.width * M.height;
    for (int i = 0; i < sz; i ++) 
        M.data[i] = ((double)rand() / ((double)RAND_MAX / (RAND_UB - RAND_LB))) + RAND_LB;
}



