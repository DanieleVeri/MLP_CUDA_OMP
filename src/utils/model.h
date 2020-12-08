#ifndef __MODEL__H
#define __MODEL__H

#include <stdlib.h>
#include <math.h>
#include "../constants.h"

// floating point precision used (float32 - IEEE 754)
typedef float fp;

// initialization types
typedef enum {
    ZERO, RAND_UNIFORM
} init_t;

// vector data struct
typedef struct {
    unsigned int len;
    fp* data;
} vector_t;

vector_t new_vector(unsigned int length, init_t init_type);
void free_vector(vector_t vector);

// matrix data struct
typedef struct {
    unsigned int m;
    unsigned int n;
    fp** data;
} matrix_t;

matrix_t new_matrix(unsigned int m, unsigned int n, init_t init_type);
void free_matrix(matrix_t matrix);

// model
typedef struct {
    unsigned int num_layer;
    matrix_t* weights_list;
    vector_t* bias_list;
} model_t;

model_t new_model(unsigned int inputs, unsigned int num_layer, init_t init_type);
void free_model(model_t layers);

// activation functions
#define ID(x) (x)
#define RELU(x) ((x > 0) * x) // branchless RELU
#define SIGMOID(x) ((fp) 1/(1+exp(-(double) x)))

#define ACTIVATION RELU

#endif