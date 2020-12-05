#ifndef __IO__H
#define __IO__H

#include <stdio.h>
#include "math.h"

void print_vector(vector_t vec) {
    printf("vector: %d\n", vec.len);
    for (int i=0; i<vec.len; i++) {
        printf("%f\n", vec.data[i]);
    }
}

void print_matrix(matrix_t mat) {
    printf("matrix: %d x %d\n", mat.m, mat.n);
    for (int i=0; i<mat.m; i++) {
        for (int j=0; j<mat.n; j++) {
            printf("%f\t", mat.data[i][j]);
        }
        printf("\n");
    }
}

void print_layers(layers_t layers) {
    printf("layers: %d\n", layers.num_layer);
    for (int i=0; i<layers.num_layer; i++) {
        printf("\tweights"); print_matrix(layers.weights_list[i]);
        printf("\tbias"); print_vector(layers.bias_list[i]);
    }
}

#endif