#include "openmp.h"

static inline float id_tf(float x) { return x; }

static inline float relu_tf(float x) { return (x > 0) * x; }

static inline float sigm_tf(float x) { return (float) 1/(1+exp(-(double) x)); }

vector_t forward_mlp(vector_t in, layers_t layers)
{
    #pragma omp parallel
    {
        int my_rank = omp_get_thread_num();
        int thread_count = omp_get_num_threads();
        printf("Hello from thread %d of %d\n", my_rank, thread_count);
    }
}