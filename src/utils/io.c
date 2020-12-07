#include "io.h"

void print_vector(vector_t vec) {
    printf("vector: %d\n", vec.len);
    for (unsigned int i=0; i<vec.len; i++) {
        printf("%f\n", vec.data[i]);
    }
}

void print_matrix(matrix_t mat) {
    printf("matrix: %d x %d\n", mat.m, mat.n);
    for (unsigned int i=0; i<mat.m; i++) {
        for (unsigned int j=0; j<mat.n; j++) {
            printf("%f\t", mat.data[i][j]);
        }
        printf("\n");
    }
}

void print_layers(layers_t layers) {
    printf("layers: %d\n", layers.num_layer);
    for (unsigned int i=0; i<layers.num_layer; i++) {
        printf("\tweights"); print_matrix(layers.weights_list[i]);
        printf("\tbias"); print_vector(layers.bias_list[i]);
    }
}