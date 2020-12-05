#include "openmp.h"

vector_t forward_mlp(vector_t in, layers_t layers)
{
    #pragma omp parallel
    {
        int my_rank = omp_get_thread_num();
        int thread_count = omp_get_num_threads();
        printf("Hello from thread %d of %d\n", my_rank, thread_count);
    }
}