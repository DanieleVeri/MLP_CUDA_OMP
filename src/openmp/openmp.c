#include "openmp.h"

vector_t forward_mlp(vector_t in, layers_t layers)
{
    const int my_rank = omp_get_thread_num();
    const int thread_count = omp_get_num_threads();
    printf("Hello from thread %d of %d\n", my_rank, thread_count);
    vector_t a;
    return a;
}