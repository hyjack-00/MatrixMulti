#include "matrix_def.h"
#include "MM_tile.h"
#include "MM_neon.h"

#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <chrono>

#include <queue>
#include <functional>  // greater<>

#include <fstream>

using namespace std;

#define LOOP_S  10
#define LOOP_M  50
#define LOOP_L  100

#define RAND_SEED1 20221124
#define RAND_SEED2 20221123
#define RAND_UB 10  // [LB, UB)
#define RAND_LB 0

// 文件IO
ofstream ofs;
string ouput_file = "../output/output1.txt";

// 计时简化
/* 
    auto start = Now;
    //...
    auto end = Now;
    cout << Dur(start, end) << endl;
*/
#define Now std::chrono::system_clock::now()
#define Dur(start,end) static_cast<std::chrono::duration<double>>((end)-(start)).count()

// Random init
template <typename T>
void rand_mat_1D(Mat_1D<T> &M, unsigned int seed) {
}
template <>
void rand_mat_1D(Mat_1D<int> &M, unsigned int seed) {
    srand(seed);
    int sz = M.size;
    for (int i = 0; i < sz; i ++) 
        for (int j = 0; j < sz; j ++)
            M.data[i*sz+j] = (rand() % (RAND_UB - RAND_LB)) + RAND_LB;
}

template <typename T>
void rand_mat_2C(Mat_2C<T> &M, unsigned int seed) {
}
template <>
void rand_mat_2C(Mat_2C<int> &M, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < M.size; i ++)
        for (int j = 0; j < M.size; j ++)
            M.data[i][j] = (rand() % (RAND_UB - RAND_LB)) + RAND_LB;
}


// Unit test -----------------------------------------------------------

void test_tile_precise() {
    ostream &os = ofs;
    os << "Precise tiling test:" << endl;
    constexpr int loop = 10, size = 2048;
    Mat_1D<int> A(size), B(size), C(size);
    rand_mat_1D(A, RAND_SEED1);
    rand_mat_1D(B, RAND_SEED2);
    os << loop << size << endl;
}

void test_tile_reg(double &reg, double &no_reg) {
    // 结果：几乎无区别
    cout << "Tiling & register test:" << endl;
    constexpr int loop = 1000, size = 4096, Tsize = 64;
    Mat_1D<int> A(size), B(size), C(size);
    rand_mat_1D(A, RAND_SEED1);
    rand_mat_1D(B, RAND_SEED2);

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

struct Rec_tile {
    int Ti, Tj, Tk;
    float time;
    Rec_tile(int i, int j, int k, float t) : Ti(i), Tj(j), Tk(k), time(t) {}
};
bool operator<(const Rec_tile &r1, const Rec_tile &r2) {
    return r1.time < r2.time;
}
void test_tile() {
    cout << "Tiling test:" << endl;
    constexpr int loop = 100, size = 4096;
    Mat_1D<int> A(size), B(size), C(size);

    // cout << "  Ti   Tj   Tk   Time" << endl;
    
    priority_queue<Rec_tile> q;
    for (int x = 1; x <= 20; x ++) q.push(Rec_tile(0, 0, 0, 100));  // 选取时间最少的前20

    for (int Ti = 8; Ti <= 1024; Ti *= 2) {
        for (int Tj = 8; Tj <= 64; Tj *= 2) {
            for (int Tk = 8; Tk <= 64; Tk *= 2) {
                // cout << setw(4) << Ti << " " << setw(4) << Tj << " " << setw(4) << Tk << "   ";
                auto start = Now;
                for (int l = 0; l < loop; l ++) {
                    memset(C.data, 0, sizeof(int)*size*size);
                    mm_1D_tile_2pow(A.data, B.data, C.data, size, Ti, Tj, Tk);
                }
                auto end = Now;
                double dur = Dur(start, end);
                // cout << dur;
                if (dur < q.top().time) {  // 进入前20
                    q.pop();
                    q.push(Rec_tile(Ti, Tj, Tk, dur));
                    // cout << " recorded";
                }
                // cout << endl;
            }
        }
    }

    // 展示前20
    cout << endl << "========== Speedest 20 ==========" << endl;
    cout << "  Ti   Tj   Tk   Time" << endl;
    while (!q.empty()) {
        Rec_tile r = q.top();
        q.pop();
        cout << setw(4) << r.Ti << " " << setw(4) << r.Tj << " " << setw(4) << r.Tk << "   ";
        cout << r.time << endl;
    }
}

void test_mat_access_speed() {
    constexpr int loop = 100, size = 4096;
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
    constexpr int size = 1024, loop = 10;
    cout << "register/restrict test:" << endl;

    Mat_1D<int> A1(size), B1(size), C1(size);
    rand_mat_1D(A1, RAND_SEED1);
    rand_mat_1D(B1, RAND_SEED2);
    auto start = Now;
    for (int i = 0; i < loop; i ++) {
        // mm_1D_benchmark_reg(A1.data, B1.data, C1.data, size);
    }
    auto end = Now;
    cout << "Mat_1D: " << Dur(start, end) << endl;

    Mat_2C<int> A2(size), B2(size), C2(size);
    rand_mat_2C(A2, RAND_SEED1);
    rand_mat_2C(B2, RAND_SEED2);
    start = Now;
    for (int i = 0; i < loop; i ++) {
        mm_2C_benchmark(A2.data, B2.data, C2.data, size);
    }
    end = Now;
    cout << "Mat_2C: " << Dur(start, end) << endl;

    cout << "register/restrict test finish." << endl;
}


int main() {
    cout << "Test begin." << endl;
    
    cout << "Output File: " << ouput_file << endl;
    ofs.open(ouput_file, ios::out);

    // double r = 0, nr = 0;
    for (int i = 0; i < 10; i ++) {
        // test_mat_access_speed();
        // test_reg_restrict();
        // test_tile_reg(r, nr);
        test_tile_precise();
    }
    cout << "Test end." << endl;
    ofs.close();
}
