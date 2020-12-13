#include "cuda.h"
#include "../utils/hpc.h"

#define ASSERT_NO_ERR(exp) {cudaSafeCall((exp)); cudaCheckError();}

__global__ void add( int *a, int *b, int *c )
{
    *c = *a + *b;
}

matrix_t cuda1_forward_mlp(matrix_t input_batch, model_t model)
{
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

    int a, b, c;	          /* host copies of a, b, c */ 
    int *d_a, *d_b, *d_c;	  /* device copies of a, b, c */
    const size_t size = sizeof(int);
    /* Allocate space for device copies of a, b, c */
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);
    /* Setup input values */
    a = 2; b = 7;
    /* Copy inputs to device */
    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);
    /* Launch add() kernel on GPU */
    add<<<1,1>>>(d_a, d_b, d_c);
    /* Copy result back to host */
    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);
    /* check result */
    if ( c != a + b ) {
        fprintf(stderr, "Test FAILED: expected %d, got %d\n", a+b, c);
    } else {
        printf("Test OK\n");
    }
    /* Cleanup */
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
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
    ASSERT_NO_ERR(cudaMemcpy(&(d_mat->m), &(mat->m), sizeof(int), cudaMemcpyHostToDevice));
    ASSERT_NO_ERR(cudaMemcpy(&(d_mat->n), &(mat->n), sizeof(int), cudaMemcpyHostToDevice));
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