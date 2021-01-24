#ifndef __MODEL__H
#define __MODEL__H

#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "../constants.h"

// initialization types
typedef enum {
    ZERO, RAND_UNIFORM
} init_t;

// vector data struct
typedef struct {
    unsigned int len;
    float* data;
} vector_s;
typedef vector_s* vector_t;

vector_t new_vector(unsigned int length, init_t init_type);
void free_vector(vector_t vector);
void assert_equal_vector(vector_t v1, vector_t v2);

// matrix data struct
typedef struct {
    unsigned int m;
    unsigned int n;
    float** data;
} matrix_s;
typedef matrix_s* matrix_t;

matrix_t new_matrix(unsigned int m, unsigned int n, init_t init_type);
void free_matrix(matrix_t matrix);
void assert_equal_matrix(matrix_t m1, matrix_t m2);

// model
typedef struct {
    unsigned int num_layer;
    matrix_t* weights_list;
    vector_t* bias_list;
} model_s;
typedef model_s* model_t;

model_t new_model(unsigned int inputs, unsigned int num_layer, init_t init_type);
void free_model(model_t layers);
matrix_t serial_forward_mlp(matrix_t input_batch, model_t model);

// branchless RELU
#define RELU(x) ((x) * ((x) > 0))
#define SIGM(x) (1/(1+exp(-(x))))
#define ACTIVATION SIGM

#define EPS 1e-5

#endif