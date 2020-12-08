#include "model.h"

vector_t new_vector(unsigned int length, init_t init_type) 
{
    vector_t vec;
    vec.len = length;
    vec.data = (fp*) malloc(length * sizeof(fp));
    for (unsigned int i=0; i<length; i++) {
        switch (init_type)
        {
        case RAND_UNIFORM:
            vec.data[i] = (fp) rand() / (fp) RAND_MAX - 0.5;
            break;
        default:
            vec.data[i] = 0;
            break;
        }
        
    }
    return vec;
}

void free_vector(vector_t vector) 
{
    free(vector.data);
}

matrix_t new_matrix(unsigned int m, unsigned int n, init_t init_type) 
{
    matrix_t mat;
    mat.m = m; mat.n = n;
    mat.data = (fp**) malloc(m * sizeof(fp*));
    for (unsigned int i=0; i<m; i++) {
        mat.data[i] = (fp*) malloc(n * sizeof(fp));
        for (unsigned int j=0; j<n; j++) {
            switch (init_type)
            {
            case RAND_UNIFORM:
                mat.data[i][j] = (fp) rand() / (fp) RAND_MAX - 0.5;
                break;
            default:
                mat.data[i][j] = 0;
                break;
            }
        }
    }
    return mat;
}

void free_matrix(matrix_t matrix)
{
    for (unsigned int i=0; i<matrix.m; i++) {
        free(matrix.data[i]);
    }
    free(matrix.data);
}

model_t new_model(unsigned int inputs, unsigned int num_layer, init_t init_type) {
    model_t obj;
    obj.num_layer = num_layer;
    obj.weights_list = (matrix_t*) malloc(num_layer * sizeof(matrix_t));
    obj.bias_list = (vector_t*) malloc(num_layer * sizeof(vector_t));
    unsigned int last = inputs;
    for (unsigned int i=0; i<num_layer; i++) {
        obj.weights_list[i] = new_matrix(last-(R-1), R, init_type);
        obj.bias_list[i] = new_vector(last-(R-1), init_type);
        last -= R-1;
    }
    return obj;
}

void free_model(model_t model) {
    for (unsigned int i=0; i<model.num_layer; i++) {
        free_matrix(model.weights_list[i]);
        free_vector(model.bias_list[i]);
    }
    free(model.weights_list);
    free(model.bias_list);
}