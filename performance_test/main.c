#include <time.h>
#include <stdio.h>
#include <omp.h>

#define WARMUP_LOOP (1e9)
#define LOOP (1e9)

typedef void (*kernel_func_t)(int); 

void kernel_1(int);  // f32(12 instr)
void kernel_2(int);  // f32
void kernel_3(int);  // f64
void kernel_4(int);  // s32

#define OP_PER_LOOP_K1 (12 * 2 * 4)
#define OP_PER_LOOP_K2 (32 * 2 * 4)
#define OP_PER_LOOP_K3 (32 * 2 * 2)
#define OP_PER_LOOP_K4 (32 * 2 * 4)

static double get_time(struct timespec *start, struct timespec *end) {
    return end->tv_sec - start->tv_sec + (end->tv_nsec - start->tv_nsec) * 1e-9;
}

static void test_parallel(kernel_func_t kernel_func,
                             const bool isFloat,
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
    if (isFloat)
        printf("perf: %.6lf GFlOPS\r\n", 
            (double)threads_num * loop * ops_per_loop * 1e-9 / time_used);
    else
        printf("perf: %.6lf DMIPS\r\n",
            (double)threads_num * loop * ops_per_loop * 1e-6 / time_used);
}

static void test_single(kernel_func_t kernel_func, 
                           const bool isFloat,
                            const int loop,
                            const int ops_per_loop) {
    struct timespec start, end;
    double time_used = 0.0;
    
    kernel_func(WARMUP_LOOP);

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    kernel_func(loop);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    time_used = get_time(&start, &end);
    if (isFloat)
        printf("perf: %.6lf GFlOPS\r\n", 
            (double)loop * ops_per_loop * 1e-9 / time_used);
    else
        printf("perf: %.6lf DMIPS\r\n",
            (double)loop * ops_per_loop * 1e-6 / time_used);
}

int main() {
    int threads_num = omp_get_max_threads();

    #ifndef __ARM_NEON
        printf("No NEON !!");
        return 0;
    #endif

    printf("Neon test:\n");
    
    printf("fp32-kernel_1 paral\n");
    test_parallel(kernel_1, threads_num, LOOP, OP_PER_LOOP_K1, 1);
    printf("fp32-kernel_1 single\n\n");
    test_single(kernel_1, LOOP, OP_PER_LOOP_K1, 1);

    printf("fp32-kernel_2 paral\n");
    test_parallel(kernel_2, threads_num, LOOP, OP_PER_LOOP_K2, 1);
    printf("fp32-kernel_2 single\n\n");
    test_single(kernel_2, LOOP, OP_PER_LOOP_K2, 1);

    printf("fp64-kernel_3 paral\n");
    test_parallel(kernel_3, threads_num, LOOP, OP_PER_LOOP_K3, 1);
    printf("fp64-kernel_3 single\n");
    test_single(kernel_3, LOOP, OP_PER_LOOP_K3, 1);

    printf("int32-kernel_4 paral\n");
    test_parallel(kernel_4, threads_num, LOOP, OP_PER_LOOP_K4, 0);
    printf("int32-kernel_4 single\n\n");
    test_single(kernel_4, LOOP, OP_PER_LOOP_K4, 0);

    return 0;
}
