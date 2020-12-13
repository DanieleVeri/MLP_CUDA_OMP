#include "log.h"

void print_vector(vector_t vec) {
    printf("vector: %d\n", vec->len);
    for (unsigned int i=0; i<vec->len; i++) {
        printf("%f\n", vec->data[i]);
    }
}

void print_matrix(matrix_t mat) {
    printf("matrix: %d x %d\n", mat->m, mat->n);
    for (unsigned int i=0; i<mat->m; i++) {
        for (unsigned int j=0; j<mat->n; j++) {
            printf("%f\t", mat->data[i][j]);
        }
        printf("\n");
    }
}

void print_model(model_t model) {
    printf("layers: %d\n", model->num_layer);
    for (unsigned int i=0; i<model->num_layer; i++) {
        printf("\tweights "); print_matrix(model->weights_list[i]);
        printf("\tbias "); print_vector(model->bias_list[i]);
    }
}