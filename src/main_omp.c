#include "utils/hpc.h"
#include "constants.h"
#include "utils/model.h"
#include "utils/io.h"
#include "openmp/openmp.h"
#include <stdio.h>
#include <assert.h>

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
    
    srand(SEED);

    vector_t input = new_vector(N, RAND_UNIFORM);
    model_t model = new_model(N, K, RAND_UNIFORM);

    double tstart = hpc_gettime();
    vector_t output = serial_forward_mlp(input, model);
    double tstop = hpc_gettime();

    printf("elapsed time = %f s\n\n", tstop - tstart);
    print_vector(output);

    free_vector(input);
    free_model(model);
    free_vector(output);

    return EXIT_SUCCESS;
}