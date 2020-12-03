#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "constants.h"
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

    #pragma omp parallel num_threads(10)
        say_hello();

    return EXIT_SUCCESS;
}