#include "utils/hpc.h"
#include "constants.h"
#include "utils/model.h"
#include "utils/log.h"
#include "cuda/cuda.h"
#include <stdio.h>

int main(int argc, char** argv)
{
    if (argc != 3) {
        printf("Required two arguments: N and K\n");
        return EXIT_FAILURE;
    }
    const int N = atoi(argv[1]);
    const int K = atoi(argv[2]);
    printf("\nMLP CUDA implementation invoked with: \n\
        N=%d (input length)\n\
        K=%d (layers)\n\
        R=%d (window size)\n\
        B=%d (batch size)\n\n", N, K, R, BATCH_SIZE);
    if (N - K*(R-1) <= 0) {
        printf("ERROR: illegal parameters: must be N > K*(%d-1)\n", R);
        return EXIT_FAILURE;
    }
    
    srand(SEED);

    //test_device_mem_leak();

    matrix_t input_batch = new_matrix(BATCH_SIZE, N, RAND_UNIFORM);
    model_t model = new_model(N, K, RAND_UNIFORM);
    printf("Created input batch\n\n");

    // serial
    double tstart = hpc_gettime();
    matrix_t output_serial = serial_forward_mlp(input_batch, model);
    double tstop = hpc_gettime();
    printf("Serial time elapsed = %f s\n\n", tstop - tstart);

    // p1
    tstart = hpc_gettime();
    matrix_t output_p1 = cuda1_forward_mlp(input_batch, model);
    tstop = hpc_gettime();
    printf("P1 time elapsed = %f s\n\n", tstop - tstart);
    
    assert_equal_matrix(output_serial, output_p1);
    printf("Test OK\n");

    free_matrix(output_serial);
    free_matrix(output_p1);
    free_matrix(input_batch);
    free_model(model);

    return EXIT_SUCCESS;
}