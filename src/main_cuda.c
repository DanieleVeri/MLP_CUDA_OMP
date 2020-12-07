#include "utils/hpc.h"
#include "utils/model.h"
#include "utils/io.h"
#include "cuda/cuda.h"
#include <stdio.h>

int main(int argc, char** argv) 
{
    hello_cuda();
    vector_t in = init_vec_uniform(10);
    print_vector(in);
    return EXIT_SUCCESS;
}
