#include "math.h"

vector_t init_vec_uniform(unsigned int length) 
{
    srand(SEED);
    vector_t vec;
    vec.len = length;
    vec.data = (float*) malloc(length * sizeof(float));
    for (int i=0; i<length; i++) {
        vec.data[i] = (float) rand() / (float) RAND_MAX;
    }
    return vec;
}

void free_vector(vector_t vector) 
{
    free(vector.data);
}

matrix_t init_mat_uniform(unsigned int m, unsigned int n) 
{
    srand(SEED);
    matrix_t mat;
    mat.m = m; mat.n = n;
    mat.data = (float**) malloc(m * sizeof(float*));
    for (int i=0; i<m; i++) {
        mat.data[i] = (float*) malloc(n * sizeof(float));
        for (int j=0; j<n; j++) {
            mat.data[i][j] = (float) rand() / (float) RAND_MAX;
        }
    }
    return mat;
}

void free_matrix(matrix_t matrix)
{
    for (int i=0; i<matrix.m; i++) {
        free(matrix.data[i]);
    }
    free(matrix.data);
}

layers_t init_layers_uniform(unsigned int inputs, unsigned int num_layer) {
    layers_t obj;
    obj.num_layer = num_layer;
    obj.weights_list = (matrix_t*) malloc(num_layer * sizeof(matrix_t));
    obj.bias_list = (vector_t*) malloc(num_layer * sizeof(vector_t));
    unsigned int last = inputs;
    for (int i=0; i<num_layer; i++) {
        obj.weights_list[i] = init_mat_uniform(last, last-(R-1));
        obj.bias_list[i] = init_vec_uniform(last-(R-1));
        last -= R-1;
    }
    return obj;
}

void free_layers(layers_t layers) {
    for (int i=0; i<layers.num_layer; i++) {
        free_matrix(layers.weights_list[i]);
        free_vector(layers.bias_list[i]);
    }
    free(layers.weights_list);
    free(layers.bias_list);
}