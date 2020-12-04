#ifndef __INPUT__H
#define __INPUT__H

#include <stdlib.h>
#include "constants.h"

typedef const float** matrix_t;
typedef const float* vector_t;

vector_t vec_rand_uniform(unsigned int length) 
{
    srand(SEED);
    float* vec = (float*) malloc(length * sizeof(float));
    for (int i=0; i<length; i++) {
        vec[i] = (float) rand() / (float) RAND_MAX;
    }
    return vec;
}

matrix_t mat_rand_uniform(unsigned int m, unsigned int n) 
{
    srand(SEED);
    float** mat = (float**) malloc(m * sizeof(float*));
    for (int i=0; i<m; i++) {
        mat[i] = (float*) malloc(n * sizeof(float));
        for (int j=0; j<n; j++) {
            mat[i][j] = (float) rand() / (float) RAND_MAX;
        }
    }
    return (matrix_t) mat;
}

#endif