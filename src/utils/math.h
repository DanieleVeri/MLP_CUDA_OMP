#ifndef __MATH_UTILS__H
#define __MATH_UTILS__H

#include <stdlib.h>
#include "../constants.h"

typedef struct {
    int len;
    float* data;
} vector_t;

typedef struct {
    int m;
    int n;
    float** data;
} matrix_t;

typedef struct {
    unsigned int num_layer;
    matrix_t* weights_list;
    vector_t* bias_list;
} layers_t;

vector_t init_vec_uniform(unsigned int length);
void free_vector(vector_t vector);

matrix_t init_mat_uniform(unsigned int m, unsigned int n);
void free_matrix(matrix_t matrix);

layers_t init_layers_uniform(unsigned int inputs, unsigned int num_layer);
void free_layers(layers_t layers);


#endif