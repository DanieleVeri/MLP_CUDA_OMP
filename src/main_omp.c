#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "constants.h"
#include "math_utils.h"
#include "hpc.h"

void say_hello(void)
{
    const int my_rank = omp_get_thread_num();
    const int thread_count = omp_get_num_threads();
    printf("Hello from thread %d of %d\n", my_rank, thread_count);
}

int main(int argc, char** argv)
{
    if (argc != 3) {
        printf("Required two arguments: N and K\n");
        return EXIT_FAILURE;
    }
    
    const int N = atoi(argv[1]);
    const int K = atoi(argv[2]);
    printf("Invoked with N=%d, K=%d, R=%d\n", N, K, R);

    printf("Input: random uniform\n");
    vector_t input = vec_rand_uniform(N);
    for (int i=0; i<N; i++) {
        printf("%f\n", input[i]);
    }
    printf("Weights init: random uniform\n");
    matrix_t weights = mat_rand_uniform(N, N-(R-1));
    for (int i=0; i<N; i++) {
        for (int j=0; j<N-(R-1); j++) {
            printf("%f\t", weights[i][j]);
        }
        printf("\n");
    }

    #pragma omp parallel num_threads(10) default(none)
        say_hello();

    free((void*) input);
    free((void*) weights);

    return EXIT_SUCCESS;
}