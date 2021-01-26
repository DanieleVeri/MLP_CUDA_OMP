#include "cuda.h"
#include "../utils/hpc.h"

#define BLKDIM 1024

__global__ void kernel_layer(matrix_t in, model_t mdl, matrix_t out, int* layer)
{
    __shared__ float temp_i[BLKDIM+R];
    const unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= out->n)
        return;

    const float** w = (const float**) mdl->weights_list[*layer]->data;
    const float* b = (const float*) mdl->bias_list[*layer]->data;

    temp_i[threadIdx.x] = in->data[0][index];
    if (threadIdx.x < R) {
        int missing = out->n - index;
        if (missing < BLKDIM)
            temp_i[2*threadIdx.x+missing] = in->data[0][threadIdx.x+out->n];
        else
            temp_i[threadIdx.x+blockDim.x] = in->data[0][index+blockDim.x];
    }
    __syncthreads();

    for (unsigned int bn=0; bn<in->m; bn++) { //for each batch
        float sum = b[index];
        for (unsigned int k=0; k<R; k++) {
            sum += w[index][k] * temp_i[threadIdx.x+k];
        }
        out->data[bn][index] = (*layer != mdl->num_layer-1) ? ACTIVATION(sum) : sum;
    }
}

__host__ matrix_t cuda1_forward_mlp(matrix_t input_batch, model_t model)
{
    matrix_t d_in = h2d_matrix(input_batch);
    model_t d_mdl = h2d_model(model);

    matrix_t h_out = new_matrix(input_batch->m, input_batch->n, ZERO);
    matrix_t d_out = h2d_matrix(h_out);
    free_matrix(h_out);
    
    const unsigned num_block = (input_batch->n + BLKDIM-1)/BLKDIM;
    const unsigned num_thread = BLKDIM;
    
    double tstart = hpc_gettime();

    int *d_layer;
    cudaSafeCall(cudaMalloc((void**)&d_layer, sizeof(int)));

    for (int layer=0; layer<model->num_layer; layer++) {
        cudaSafeCall(cudaMemcpy(d_layer, &layer, sizeof(int), cudaMemcpyHostToDevice));
        cudaSafeCall(cudaMemcpy(&(d_out->n), &(model->bias_list[layer]->len), sizeof(int), cudaMemcpyHostToDevice));

        kernel_layer<<<num_block, num_thread>>>(d_in, d_mdl, d_out, d_layer);
        cudaCheckError();
        
        if (layer < model->num_layer-1) {
            matrix_t swap;
            swap = d_in;
            d_in = d_out;
            d_out = swap;
        }
    }

    cudaSafeCall(cudaFree(d_layer));

    double tstop = hpc_gettime();
    printf("P1 kernel time elapsed = %f\n", tstop-tstart);
    h_out = d2h_matrix(d_out);

    device_free_matrix(d_in);
    device_free_model(d_mdl);
    device_free_matrix(d_out);
    return h_out;
}

__host__ vector_t h2d_vector(vector_t vec) {
    vector_t d_vec;
    cudaSafeCall(cudaMalloc((void**)&d_vec, sizeof(vector_s)));
    cudaSafeCall(cudaMemcpy(&(d_vec->len), &(vec->len), sizeof(int), cudaMemcpyHostToDevice));
    float* d_data;
    cudaSafeCall(cudaMalloc((void**)&d_data, vec->len*sizeof(float)));
    cudaSafeCall(cudaMemcpy(d_data, vec->data, vec->len*sizeof(float), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(&d_vec->data, &d_data, sizeof(float*), cudaMemcpyHostToDevice));
    return d_vec;
}

__host__ vector_t d2h_vector(vector_t d_vec) {
    vector_t vec = (vector_t) malloc(sizeof(vector_s));
    cudaSafeCall(cudaMemcpy(vec, d_vec, sizeof(vector_s), cudaMemcpyDeviceToHost));
    float* d_data = (float*) malloc(vec->len*sizeof(float));
    cudaSafeCall(cudaMemcpy(d_data, vec->data, vec->len*sizeof(float), cudaMemcpyDeviceToHost));
    vec->data = d_data;
    return vec;
}

__host__ void device_free_vector(vector_t d_vec) {
    float* d_data;
    cudaSafeCall(cudaMemcpy(&d_data, &d_vec->data, sizeof(float*), cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaFree(d_data));
    cudaSafeCall(cudaFree(d_vec));
}

__host__ matrix_t h2d_matrix(matrix_t mat) {
    matrix_t d_mat;
    cudaSafeCall(cudaMalloc((void**)&d_mat, sizeof(matrix_s)));
    cudaSafeCall(cudaMemcpy(&(d_mat->m), &(mat->m), sizeof(unsigned int), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(&(d_mat->n), &(mat->n), sizeof(unsigned int), cudaMemcpyHostToDevice));
    float** d_data;
    float* d_blk;
    cudaSafeCall(cudaMalloc((void**)&d_data, mat->m*sizeof(float*)));
    cudaSafeCall(cudaMalloc((void**)&d_blk, mat->m*mat->n*sizeof(float)));
    cudaSafeCall(cudaMemcpy(d_blk, mat->data[0], mat->m*mat->n*sizeof(float), cudaMemcpyHostToDevice));
    float* addr[mat->m];
    for (unsigned int i=0; i<mat->m; i++) {
        addr[i] = &d_blk[i*mat->n];
    }
    cudaSafeCall(cudaMemcpy(d_data, addr, mat->m*sizeof(float*), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(&d_mat->data, &d_data, sizeof(float**), cudaMemcpyHostToDevice));
    return d_mat;
}

__host__ matrix_t d2h_matrix(matrix_t d_mat) {
    matrix_t mat = (matrix_t) malloc(sizeof(matrix_s));
    cudaSafeCall(cudaMemcpy(mat, d_mat, sizeof(matrix_s), cudaMemcpyDeviceToHost));
    float** data = (float**) malloc(mat->m*sizeof(float*));
    cudaSafeCall(cudaMemcpy(data, mat->data, mat->m*sizeof(float*), cudaMemcpyDeviceToHost));
    float* blk = (float*) malloc(mat->m*mat->n*sizeof(float));
    cudaSafeCall(cudaMemcpy(blk, data[0], mat->m*mat->n*sizeof(float), cudaMemcpyDeviceToHost));
    for (unsigned int i=0; i<mat->m; i++) {
        data[i] = &(blk[i*mat->n]);
    }
    mat->data = data;
    return mat;
}

__host__ void device_free_matrix(matrix_t d_mat) {
    float** d_data;
    cudaSafeCall(cudaMemcpy(&d_data, &d_mat->data, sizeof(float**), cudaMemcpyDeviceToHost));
    float *d_blk;
    cudaSafeCall(cudaMemcpy(&d_blk, d_data, sizeof(float*), cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaFree(d_blk));
    cudaSafeCall(cudaFree(d_data));
    cudaSafeCall(cudaFree(d_mat));
}

__host__ model_t h2d_model(model_t mdl) {
    model_t d_mdl;
    cudaSafeCall(cudaMalloc((void**)&d_mdl, sizeof(model_s)));
    cudaSafeCall(cudaMemcpy(&(d_mdl->num_layer), &(mdl->num_layer), sizeof(unsigned int), cudaMemcpyHostToDevice));
    matrix_t* d_weights;
    vector_t* d_biases;
    cudaSafeCall(cudaMalloc((void**)&d_weights, mdl->num_layer*sizeof(matrix_t*)));
    cudaSafeCall(cudaMalloc((void**)&d_biases, mdl->num_layer*sizeof(vector_t*)));
    for (unsigned int i=0; i<mdl->num_layer; i++) {
        matrix_t d_m = h2d_matrix(mdl->weights_list[i]);
        vector_t d_v = h2d_vector(mdl->bias_list[i]);
        cudaSafeCall(cudaMemcpy(&d_weights[i], &d_m, sizeof(matrix_t), cudaMemcpyHostToDevice));
        cudaSafeCall(cudaMemcpy(&d_biases[i], &d_v, sizeof(vector_t), cudaMemcpyHostToDevice));
    }
    cudaSafeCall(cudaMemcpy(&d_mdl->weights_list, &d_weights, sizeof(matrix_t*), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(&d_mdl->bias_list, &d_biases, sizeof(vector_t*), cudaMemcpyHostToDevice));
    return d_mdl;
}

__host__ void device_free_model(model_t d_mdl) {
    matrix_t* d_weights;
    vector_t* d_biases;
    unsigned int num_layer;
    cudaSafeCall(cudaMemcpy(&num_layer, &d_mdl->num_layer, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpy(&d_weights, &d_mdl->weights_list, sizeof(matrix_t*), cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpy(&d_biases, &d_mdl->bias_list, sizeof(vector_t*), cudaMemcpyDeviceToHost));
    for (unsigned int i=0; i<num_layer; i++) {
        matrix_t d_w;
        vector_t d_b;
        cudaSafeCall(cudaMemcpy(&d_w, &d_weights[i], sizeof(matrix_t), cudaMemcpyDeviceToHost));
        cudaSafeCall(cudaMemcpy(&d_b, &d_biases[i], sizeof(vector_t), cudaMemcpyDeviceToHost));
        
        device_free_matrix(d_w);
        device_free_vector(d_b);
    }
    cudaSafeCall(cudaFree(d_weights));
    cudaSafeCall(cudaFree(d_biases));
    cudaSafeCall(cudaFree(d_mdl));
}

__host__ void test_device_mem_leak() {
    const unsigned int cycles = 500;
    vector_t vec = new_vector(1000, RAND_UNIFORM);
    matrix_t mat = new_matrix(1000, 1000, RAND_UNIFORM);
    model_t mdl = new_model(100, 20, RAND_UNIFORM);
    printf("Device memory leak stress test: running %d cycles...\n", cycles);

    for (unsigned int i=0; i<cycles; i++) {
        vector_t d_vec = h2d_vector(vec);
        vector_t h_vec = d2h_vector(d_vec);
        assert_equal_vector(vec, h_vec);
        free_vector(h_vec);
        device_free_vector(d_vec);

        matrix_t d_mat = h2d_matrix(mat);
        matrix_t h_mat = d2h_matrix(d_mat);
        assert_equal_matrix(mat, h_mat);
        free_matrix(h_mat);
        device_free_matrix(d_mat);
        
        model_t d_mdl = h2d_model(mdl);
        device_free_model(d_mdl);
        printf("%d  ", i); fflush(stdout);
    }
    printf("\n");
}