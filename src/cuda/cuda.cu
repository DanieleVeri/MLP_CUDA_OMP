#include "cuda.h"
#include "../utils/hpc.h"

#define BLKDIM 1024

__global__ void kernel_layer(matrix_t in, model_t mdl, matrix_t out, int* layer)
{
    // shared memory to exploit data reuse of the input
    __shared__ float temp_i[BLKDIM+R];
    // index and batch
    const unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int batch = blockIdx.y;
    // current layer weights and biases
    const float** w = (const float**) mdl->weights_list[*layer]->data;
    const float* b = (const float*) mdl->bias_list[*layer]->data;
    // current output size
    const unsigned max_n = mdl->bias_list[*layer]->len;
    if (index >= max_n)
        return;

    // fill shared mem handling the case that input is not a multiple of block size
    temp_i[threadIdx.x] = in->data[batch][index];
    if (threadIdx.x < R) {
        int missing = max_n - index;
        if (missing < BLKDIM)
            temp_i[2*threadIdx.x+missing] = in->data[batch][threadIdx.x+max_n];
        else
            temp_i[threadIdx.x+blockDim.x] = in->data[batch][index+blockDim.x];
    }
    __syncthreads();

    // compute neuron output
    float sum = b[index];
    for (unsigned int k=0; k<R; k++) {
        sum += w[index][k] * temp_i[threadIdx.x+k];
    }
    // apply activation if isn't the last layer (the model is a regressor)
    out->data[batch][index] = (*layer != mdl->num_layer-1) ? ACTIVATION(sum) : sum;
}

__host__ matrix_t cuda_forward_mlp(matrix_t input_batch, model_t model)
{
    // alloc and copy input and model to device memory
    matrix_t d_in = h2d_matrix(input_batch);
    model_t d_mdl = h2d_model(model);

    // create buffer matrix as big as the input (create on host, move on device, free on host)
    matrix_t h_buff = new_matrix_pinned(input_batch->m, input_batch->n, ZERO);
    matrix_t d_buff = h2d_matrix(h_buff);
    free_matrix_pinned(h_buff);
    
    // 2D grid, x is for number of features, y is for batch elements
    dim3 grid((input_batch->n + BLKDIM-1)/BLKDIM, input_batch->m);
    dim3 block(BLKDIM);
    
    // Alloc on device current layer index
    int *d_layer;  cudaSafeCall(cudaMalloc((void**)&d_layer, sizeof(int)));
    double tstart = hpc_gettime();
    // layer loop
    for (int layer=0; layer<model->num_layer; layer++) {
        // update current layer index on device
        cudaSafeCall(cudaMemcpy(d_layer, &layer, sizeof(int), cudaMemcpyHostToDevice));

        kernel_layer<<<grid, block>>>(d_in, d_mdl, d_buff, d_layer);
        cudaDeviceSynchronize();
        cudaCheckError();
        
        // swap input and output buffer pointers
        if (layer < model->num_layer-1) {
            matrix_t swap;
            swap = d_in;
            d_in = d_buff;
            d_buff = swap;
        }
    }
    double tstop = hpc_gettime();
    printf("Kernel time elapsed = %f s\n", tstop-tstart);
    cudaSafeCall(cudaFree(d_layer));

    // copy the result on host memory
    matrix_t h_out = d2h_matrix(d_buff);
    // truncate to the output dimension (it was as big as the input)
    h_out->n = model->bias_list[model->num_layer-1]->len;

    device_free_matrix(d_in);
    device_free_model(d_mdl);
    device_free_matrix(d_buff);
    return h_out;
}

/********************************** VECTOR MEMORY UTILITIES **********************************/

__host__ vector_t new_vector_pinned(unsigned int length, init_t init_type) 
{
    vector_t vec;
    cudaSafeCall(cudaMallocHost(&vec, sizeof(vector_s)));
    vec->len = length;
    cudaSafeCall(cudaMallocHost(&vec->data, length * sizeof(float)));
    for (unsigned int i=0; i<length; i++) {
        switch (init_type)
        {
        case RAND_UNIFORM: // [-2, +2)
            vec->data[i] = 4 * ((float) rand() / (float) RAND_MAX - 0.5);
            break;
        default:
            vec->data[i] = 0;
            break;
        }
        
    }
    return vec;
}

__host__ void free_vector_pinned(vector_t vector) 
{
    cudaSafeCall(cudaFreeHost(vector->data));
    cudaSafeCall(cudaFreeHost(vector));
}

__host__ vector_t h2d_vector(vector_t vec) {
    // alloc the struct
    vector_t d_vec;
    cudaSafeCall(cudaMalloc((void**)&d_vec, sizeof(vector_s)));
    cudaSafeCall(cudaMemcpy(&(d_vec->len), &(vec->len), sizeof(int), cudaMemcpyHostToDevice));
    // alloc the data, then update the vec->data pointer with that one on the device memory
    float* d_data;
    cudaSafeCall(cudaMalloc((void**)&d_data, vec->len*sizeof(float)));
    cudaSafeCall(cudaMemcpy(d_data, vec->data, vec->len*sizeof(float), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(&d_vec->data, &d_data, sizeof(float*), cudaMemcpyHostToDevice));
    return d_vec;
}

__host__ vector_t d2h_vector(vector_t d_vec) {
    // copy on host the struct
    vector_t vec;
    cudaSafeCall(cudaMallocHost(&vec, sizeof(vector_s)));
    cudaSafeCall(cudaMemcpy(vec, d_vec, sizeof(vector_s), cudaMemcpyDeviceToHost));
    // copy on host the data
    float* d_data;
    cudaSafeCall(cudaMallocHost(&d_data, vec->len*sizeof(float)));
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

/********************************** MATRIX MEMORY UTILITIES **********************************/

__host__ matrix_t new_matrix_pinned(unsigned int m, unsigned int n, init_t init_type) 
{
    matrix_t mat;
    cudaSafeCall(cudaMallocHost(&mat, sizeof(vector_s)));
    mat->m = m; mat->n = n;
    // data block (contiguous)
    float* blk;
    cudaSafeCall(cudaMallocHost(&blk, m * n * sizeof(float)));
    cudaSafeCall(cudaMallocHost(&mat->data, m * sizeof(float*)));
    // updates the elements of data with the addresses of data block
    // allowing to preserve the double indexing of the array
    for (unsigned int i=0; i<m; i++) {
        mat->data[i] = &(blk[i*n]);
        for (unsigned int j=0; j<n; j++) {
            switch (init_type)
            {
            case RAND_UNIFORM: // [-2, +2)
                mat->data[i][j] = 4 * ((float) rand() / (float) RAND_MAX - 0.5);
                break;
            default:
                mat->data[i][j] = 0;
                break;
            }
        }
    }
    return mat;
}

__host__ void free_matrix_pinned(matrix_t matrix)
{
    cudaSafeCall(cudaFreeHost(matrix->data[0])); //free blk
    cudaSafeCall(cudaFreeHost(matrix->data));
    cudaSafeCall(cudaFreeHost(matrix));
}

__host__ matrix_t h2d_matrix(matrix_t mat) {
    // copy on device the struct
    matrix_t d_mat;
    cudaSafeCall(cudaMalloc((void**)&d_mat, sizeof(matrix_s)));
    cudaSafeCall(cudaMemcpy(&(d_mat->m), &(mat->m), sizeof(unsigned int), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(&(d_mat->n), &(mat->n), sizeof(unsigned int), cudaMemcpyHostToDevice));
    // copy on device the data (d_blk)
    float** d_data;
    float* d_blk;
    cudaSafeCall(cudaMalloc((void**)&d_data, mat->m*sizeof(float*)));
    cudaSafeCall(cudaMalloc((void**)&d_blk, mat->m*mat->n*sizeof(float)));
    cudaSafeCall(cudaMemcpy(d_blk, mat->data[0], mat->m*mat->n*sizeof(float), cudaMemcpyHostToDevice));
    // update the d_data indeces with that ones od d_blk on the device memory, so to preserve the double index access
    float* addr[mat->m];
    for (unsigned int i=0; i<mat->m; i++) {
        addr[i] = &d_blk[i*mat->n];
    }
    cudaSafeCall(cudaMemcpy(d_data, addr, mat->m*sizeof(float*), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(&d_mat->data, &d_data, sizeof(float**), cudaMemcpyHostToDevice));
    return d_mat;
}

__host__ matrix_t d2h_matrix(matrix_t d_mat) {
    // copy struct to host
    matrix_t mat;
    cudaSafeCall(cudaMallocHost(&mat, sizeof(matrix_s)));
    cudaSafeCall(cudaMemcpy(mat, d_mat, sizeof(matrix_s), cudaMemcpyDeviceToHost));
    // copy data to host (data[0] will contain the memory block address)
    float** data;
    cudaSafeCall(cudaMallocHost(&data, mat->m*sizeof(float*)));
    cudaSafeCall(cudaMemcpy(data, mat->data, mat->m*sizeof(float*), cudaMemcpyDeviceToHost));
    // copy the memory block to host
    float* blk;
    cudaSafeCall(cudaMallocHost(&blk, mat->m*mat->n*sizeof(float)));
    cudaSafeCall(cudaMemcpy(blk, data[0], mat->m*mat->n*sizeof(float), cudaMemcpyDeviceToHost));
    // update data indeces with the host copy of blk
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

/********************************** MODEL MEMORY UTILITIES **********************************/

__host__ model_t new_model_pinned(unsigned int inputs, unsigned int num_layer, init_t init_type) {
    model_t obj;
    cudaSafeCall(cudaMallocHost(&obj, sizeof(model_s)));
    obj->num_layer = num_layer;
    cudaSafeCall(cudaMallocHost(&obj->weights_list, num_layer * sizeof(matrix_t)));
    cudaSafeCall(cudaMallocHost(&obj->bias_list, num_layer * sizeof(vector_t)));
    unsigned int last = inputs;
    for (unsigned int i=0; i<num_layer; i++) {
        obj->weights_list[i] = new_matrix_pinned(last-(R-1), R, init_type);
        obj->bias_list[i] = new_vector_pinned(last-(R-1), init_type);
        last -= R-1;
    }
    return obj;
}

__host__ void free_model_pinned(model_t model) {
    for (unsigned int i=0; i<model->num_layer; i++) {
        free_matrix_pinned(model->weights_list[i]);
        free_vector_pinned(model->bias_list[i]);
    }
    cudaSafeCall(cudaFreeHost(model->weights_list));
    cudaSafeCall(cudaFreeHost(model->bias_list));
    cudaSafeCall(cudaFreeHost(model));
}

__host__ model_t h2d_model(model_t mdl) {
    // copy struct to device
    model_t d_mdl;
    cudaSafeCall(cudaMalloc((void**)&d_mdl, sizeof(model_s)));
    cudaSafeCall(cudaMemcpy(&(d_mdl->num_layer), &(mdl->num_layer), sizeof(unsigned int), cudaMemcpyHostToDevice));
    // alloc the pointer lists
    matrix_t* d_weights;
    vector_t* d_biases;
    cudaSafeCall(cudaMalloc((void**)&d_weights, mdl->num_layer*sizeof(matrix_t*)));
    cudaSafeCall(cudaMalloc((void**)&d_biases, mdl->num_layer*sizeof(vector_t*)));
    // copy all weights and biases and update the lists with device pointers
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
    // copy to host the pointer lists, than free each element
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
    const unsigned int cycles = 1000;
    vector_t vec = new_vector_pinned(1000, RAND_UNIFORM);
    matrix_t mat = new_matrix_pinned(1000, 1000, RAND_UNIFORM);
    model_t mdl = new_model_pinned(100, 20, RAND_UNIFORM);
    printf("Device memory leak stress test: running %d cycles...\n", cycles);

    for (unsigned int i=0; i<cycles; i++) {
        vector_t d_vec = h2d_vector(vec);
        vector_t h_vec = d2h_vector(d_vec);
        assert_equal_vector(vec, h_vec);
        free_vector_pinned(h_vec);
        device_free_vector(d_vec);

        matrix_t d_mat = h2d_matrix(mat);
        matrix_t h_mat = d2h_matrix(d_mat);
        assert_equal_matrix(mat, h_mat);
        free_matrix_pinned(h_mat);
        device_free_matrix(d_mat);
        
        model_t d_mdl = h2d_model(mdl);
        device_free_model(d_mdl);
        printf("%d  ", i); fflush(stdout);
    }

    free_vector_pinned(vec);
    free_matrix_pinned(mat);
    free_model_pinned(mdl);
    printf("\n");
}