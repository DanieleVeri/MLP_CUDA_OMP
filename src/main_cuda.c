#include "utils/hpc.h"
#include "utils/model.h"
#include "utils/io.h"
#include "cuda/cuda.h"
#include <stdio.h>

int main(int argc, char** argv) 
{
    srand(SEED);
    hello_cuda();
    vector_t in = new_vector(10, RAND_UNIFORM);
    print_vector(in);
    return EXIT_SUCCESS;
}
