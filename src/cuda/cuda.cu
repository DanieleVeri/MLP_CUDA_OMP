#include "cuda.h"
#include "../utils/hpc.h"

#define ASSERT_NO_ERR(exp) {cudaSafeCall((exp)); cudaCheckError();}

__global__ void add( int *a, int *b, int *c )
{
    *c = *a + *b;
}

matrix_t cuda1_forward_mlp(matrix_t input_batch, model_t model)
{
    for(int i=0; i<10000; i++) {
        vector_t d_vec = h2d_vector(model->bias_list[0]);
        vector_t h_vec = d2h_vector(d_vec);
        print_vector(model->bias_list[0]);
        print_vector(h_vec);
        free_vector(h_vec);
        device_free_vector(d_vec);
        printf("vec OK\n");

        matrix_t d_mat = h2d_matrix(model->weights_list[0]);
        matrix_t h_mat = d2h_matrix(d_mat);
        print_matrix(model->weights_list[0]);
        print_matrix(h_mat);
        free_matrix(h_mat);
        device_free_matrix(d_mat);
        printf("mat OK\n");

        model_t d_mdl = h2d_model(model);
        device_free_model(d_mdl);
        printf("mdl OK\n");
    }
}

vector_t h2d_vector(vector_t vec) {
    vector_t d_vec;
    ASSERT_NO_ERR(cudaMalloc(&d_vec, sizeof(vector_s)));
    ASSERT_NO_ERR(cudaMemcpy(&(d_vec->len), &(vec->len), sizeof(int), cudaMemcpyHostToDevice));
    float* d_data;
    ASSERT_NO_ERR(cudaMalloc(&d_data, vec->len*sizeof(float)));
    ASSERT_NO_ERR(cudaMemcpy(d_data, vec->data, vec->len*sizeof(float), cudaMemcpyHostToDevice));
    ASSERT_NO_ERR(cudaMemcpy(&d_vec->data, &d_data, sizeof(float*), cudaMemcpyHostToDevice));
    return d_vec;
}

vector_t d2h_vector(vector_t d_vec) {
    vector_t vec = (vector_t) malloc(sizeof(vector_s));
    ASSERT_NO_ERR(cudaMemcpy(vec, d_vec, sizeof(vector_s), cudaMemcpyDeviceToHost));
    float* d_data = (float*) malloc(vec->len*sizeof(float));
    ASSERT_NO_ERR(cudaMemcpy(d_data, vec->data, vec->len*sizeof(float), cudaMemcpyDeviceToHost));
    vec->data = d_data;
    return vec;
}

void device_free_vector(vector_t d_vec) {
    float* d_data;
    ASSERT_NO_ERR(cudaMemcpy(&d_data, &d_vec->data, sizeof(float*), cudaMemcpyDeviceToHost));
    ASSERT_NO_ERR(cudaFree(d_data));
    ASSERT_NO_ERR(cudaFree(d_vec));
}

matrix_t h2d_matrix(matrix_t mat) {
    matrix_t d_mat;
    ASSERT_NO_ERR(cudaMalloc(&d_mat, sizeof(matrix_s)));
    ASSERT_NO_ERR(cudaMemcpy(&(d_mat->m), &(mat->m), sizeof(unsigned int), cudaMemcpyHostToDevice));
    ASSERT_NO_ERR(cudaMemcpy(&(d_mat->n), &(mat->n), sizeof(unsigned int), cudaMemcpyHostToDevice));
    float** d_data;
    ASSERT_NO_ERR(cudaMalloc(&d_data, mat->m*sizeof(float*)));
    for (unsigned int i=0; i<mat->m; i++) {
        float *d_row;
        ASSERT_NO_ERR(cudaMalloc(&d_row, mat->n*sizeof(float)));
        ASSERT_NO_ERR(cudaMemcpy(d_row, mat->data[i], mat->n*sizeof(float), cudaMemcpyHostToDevice));
        ASSERT_NO_ERR(cudaMemcpy(&d_data[i], &d_row, sizeof(float*), cudaMemcpyHostToDevice));
    }
    ASSERT_NO_ERR(cudaMemcpy(&d_mat->data, &d_data, sizeof(float**), cudaMemcpyHostToDevice));
    return d_mat;
}

matrix_t d2h_matrix(matrix_t d_mat) {
    matrix_t mat = (matrix_t) malloc(sizeof(matrix_s));
    ASSERT_NO_ERR(cudaMemcpy(mat, d_mat, sizeof(matrix_s), cudaMemcpyDeviceToHost));
    float** d_data = (float**) malloc(mat->m*sizeof(float*));
    ASSERT_NO_ERR(cudaMemcpy(d_data, mat->data, mat->m*sizeof(float*), cudaMemcpyDeviceToHost));
    for (unsigned int i=0; i<mat->m; i++) {
        float* d_row = (float*) malloc(mat->n*sizeof(float));
        ASSERT_NO_ERR(cudaMemcpy(d_row, d_data[i], mat->n*sizeof(float), cudaMemcpyDeviceToHost));
        d_data[i] = d_row;
    }
    mat->data = d_data;
    return mat;
}

void device_free_matrix(matrix_t d_mat) {
    float** d_data;
    unsigned int m;
    ASSERT_NO_ERR(cudaMemcpy(&d_data, &d_mat->data, sizeof(float**), cudaMemcpyDeviceToHost));
    ASSERT_NO_ERR(cudaMemcpy(&m, &d_mat->m, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    for (unsigned int i=0; i<m; i++) {
        float* d_row;
        ASSERT_NO_ERR(cudaMemcpy(&d_row, &d_data[i], sizeof(float*), cudaMemcpyDeviceToHost));
        ASSERT_NO_ERR(cudaFree(d_row));
    }
    ASSERT_NO_ERR(cudaFree(d_data));
    ASSERT_NO_ERR(cudaFree(d_mat));
}

model_t h2d_model(model_t mdl) {
    model_t d_mdl;
    ASSERT_NO_ERR(cudaMalloc(&d_mdl, sizeof(model_s)));
    ASSERT_NO_ERR(cudaMemcpy(&(d_mdl->num_layer), &(mdl->num_layer), sizeof(unsigned int), cudaMemcpyHostToDevice));
    matrix_t* d_weights;
    vector_t* d_biases;
    ASSERT_NO_ERR(cudaMalloc(&d_weights, mdl->num_layer*sizeof(matrix_t*)));
    ASSERT_NO_ERR(cudaMalloc(&d_biases, mdl->num_layer*sizeof(vector_t*)));
    for (unsigned int i=0; i<mdl->num_layer; i++) {
        matrix_t d_m = h2d_matrix(mdl->weights_list[i]);
        vector_t d_v = h2d_vector(mdl->bias_list[i]);
        ASSERT_NO_ERR(cudaMemcpy(&d_weights[i], &d_m, sizeof(matrix_t), cudaMemcpyHostToDevice));
        ASSERT_NO_ERR(cudaMemcpy(&d_biases[i], &d_v, sizeof(vector_t), cudaMemcpyHostToDevice));
    }
    ASSERT_NO_ERR(cudaMemcpy(&d_mdl->weights_list, &d_weights, sizeof(matrix_t*), cudaMemcpyHostToDevice));
    ASSERT_NO_ERR(cudaMemcpy(&d_mdl->bias_list, &d_biases, sizeof(vector_t*), cudaMemcpyHostToDevice));
    return d_mdl;
}

void device_free_model(model_t d_mdl) {
    matrix_t* d_weights;
    vector_t* d_biases;
    unsigned int num_layer;
    ASSERT_NO_ERR(cudaMemcpy(&num_layer, &d_mdl->num_layer, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    ASSERT_NO_ERR(cudaMemcpy(&d_weights, &d_mdl->weights_list, sizeof(matrix_t*), cudaMemcpyDeviceToHost));
    ASSERT_NO_ERR(cudaMemcpy(&d_biases, &d_mdl->bias_list, sizeof(vector_t*), cudaMemcpyDeviceToHost));
    for (unsigned int i=0; i<num_layer; i++) {
        matrix_t d_w;
        vector_t d_b;
        ASSERT_NO_ERR(cudaMemcpy(&d_w, &d_weights[i], sizeof(matrix_t), cudaMemcpyDeviceToHost));
        ASSERT_NO_ERR(cudaMemcpy(&d_b, &d_biases[i], sizeof(vector_t), cudaMemcpyDeviceToHost));
        device_free_matrix(d_w);
        device_free_vector(d_b);
    }
    ASSERT_NO_ERR(cudaFree(d_weights));
    ASSERT_NO_ERR(cudaFree(d_biases));
    ASSERT_NO_ERR(cudaFree(d_mdl));
}