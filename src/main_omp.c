#include "utils/hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "constants.h"
#include "utils/math.h"
#include "utils/io.h"
#include "openmp/openmp.h"

int main(int argc, char** argv)
{
    if (argc != 3) {
        printf("Required two arguments: N and K\n");
        return EXIT_FAILURE;
    }
    const int N = atoi(argv[1]);
    const int K = atoi(argv[2]);
    printf("Invoked with N=%d, K=%d, R=%d\n", N, K, R);
    if (N - K*(R-1) <= 0) {
        printf("ERROR: illegal parameters: must be N > K*(%d-1)\n", R);
        return EXIT_FAILURE;
    }

    vector_t input = init_vec_uniform(N);
    print_vector(input);

    layers_t layers = init_layers_uniform(N, K);
    print_layers(layers);

    vector_t output = forward_mlp(input, layers);

    free_vector(input);
    free_layers(layers);
    free_vector(output);

    return EXIT_SUCCESS;
}