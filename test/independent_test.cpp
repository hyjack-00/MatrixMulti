#include <iostream>
#include <arm_neon.h>
#include <pthread.h>
#include <chrono>
#include <stdlib.h>

struct Mat_1G_s32 {
    int32_t height, width;
    int32_t *data;

    Mat_1G(int h=0, int w=0) : width(w), height(h) {
        data = new int[w * h];
    }
    ~Mat_1G() { delete[] data; }
};

#define RAND_UB 100
#define RAND_LB -100
void rand_mat_1G_s32(Mat_1G_s32 &M, unsigned int seed) {
    srand(seed);
    int sz = M.width * M.height;
    for (int i = 0; i < sz; i ++) 
        M.data[i] = (rand() % (RAND_UB - RAND_LB)) + RAND_LB;
}


int main() {

}