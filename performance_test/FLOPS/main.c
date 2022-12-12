#include <time.h>
#include <stdio.h>
#include <omp.h>

#define WARMUP_LOOP (1e6)
#define LOOP (1e9)

typedef void (*kernel_func_t)(int); 

void kernel_1(int);  // asm implementation
#define OP_NUM_K1 (32)

void kernel_2(int);
#define OP_NUM_K2 (80)

static double get_time(struct timespec *start, struct timespec *end) {
    return end->tv_sec - start->tv_sec + (end->tv_nsec - start->tv_nsec) * 1e-9;
}

static void test_parallel(kernel_func_t kernel_func, 
                              const int threads_num,
                              const int loop,
                              const int ops_per_loop) {
    struct timespec start, end;
    double time_used = 0.0;
    
    #pragma omp parallel for
    for (int i = 0; i < threads_num; i++) {  // warmup
        kernel_func(WARMUP_LOOP);
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    #pragma omp parallel for
    for (int i = 0; i < threads_num; i++) {
        kernel_func(loop);
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    time_used = get_time(&start, &end);
    printf("perf: %.6lf GFlops\r\n", 
        (double)threads_num * loop * ops_per_loop * 1e-9 / time_used);
}

static void test_single(kernel_func_t kernel_func, 
                            const int loop,
                            const int ops_per_loop) {
    struct timespec start, end;
    double time_used = 0.0;
    
    kernel_func(WARMUP_LOOP);

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    kernel_func(loop);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    time_used = get_time(&start, &end);
    printf("perf: %.6lf GFlops\r\n", 
        (double)loop * ops_per_loop * 1e-9 / time_used);
}

int main() {
    int threads_num = omp_get_max_threads();

    #ifndef __NEON__
        printf("No NEON !!");
        return 0;
    #endif

    printf("Neon test:\n");
    printf("fp32-kernel_1 paral\n");
    test_parallel(kernel_1, threads_num, LOOP, OP_NUM_K1 * 2 * 4);

    printf("fp32-kernel_1 single\n");
    test_single(kernel_1, LOOP, OP_NUM_K1 * 2 * 4);

    return 0;
}
