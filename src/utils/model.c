#include "model.h"

vector_t new_vector(unsigned int length, init_t init_type) 
{
    vector_t vec = malloc(sizeof(vector_s));
    vec->len = length;
    vec->data = (float*) malloc(length * sizeof(float));
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

void free_vector(vector_t vector) 
{
    free(vector->data);
    free(vector);
}

void assert_equal_vector(vector_t v1, vector_t v2) 
{
    assert(v1->len == v2->len);
    for(unsigned int i=0; i<v1->len; i++) 
        assert(fabs(v1->data[i] - v2->data[i]) < EPS);
}

matrix_t new_matrix(unsigned int m, unsigned int n, init_t init_type) 
{
    matrix_t mat = malloc(sizeof(vector_s));
    mat->m = m; mat->n = n;
    // data block (contiguous)
    float* blk = (float*) malloc(m * n * sizeof(float));
    mat->data = (float**) malloc(m * sizeof(float*));
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

void free_matrix(matrix_t matrix)
{
    free(matrix->data[0]); //free blk
    free(matrix->data);
    free(matrix);
}

void assert_equal_matrix(matrix_t m1, matrix_t m2) 
{
    assert(m1->m == m2->m);
    assert(m1->n == m2->n);
    for(unsigned int i=0; i<m1->m; i++) 
        for(unsigned int j=0; j<m1->n; j++)
            assert(fabs(m1->data[i][j] - m2->data[i][j]) < EPS);
}

model_t new_model(unsigned int inputs, unsigned int num_layer, init_t init_type) 
{
    model_t obj = malloc(sizeof(model_s));
    obj->num_layer = num_layer;
    obj->weights_list = (matrix_t*) malloc(num_layer * sizeof(matrix_t));
    obj->bias_list = (vector_t*) malloc(num_layer * sizeof(vector_t));
    unsigned int last = inputs;
    for (unsigned int i=0; i<num_layer; i++) {
        obj->weights_list[i] = new_matrix(last-(R-1), R, init_type);
        obj->bias_list[i] = new_vector(last-(R-1), init_type);
        last -= R-1;
    }
    return obj;
}

void free_model(model_t model) 
{
    for (unsigned int i=0; i<model->num_layer; i++) {
        free_matrix(model->weights_list[i]);
        free_vector(model->bias_list[i]);
    }
    free(model->weights_list);
    free(model->bias_list);
    free(model);
}

matrix_t serial_forward_mlp(matrix_t input_batch, model_t model)
{
    const unsigned int batch_size = input_batch->m;
    const unsigned int last_layer = model->num_layer-1;
    matrix_t last_result = input_batch;
    // 1. layer loop (not parallelizable)
    for (unsigned int i=0; i<model->num_layer; i++) {
        const unsigned int out_len = model->bias_list[i]->len;
        const float** w = (const float**) model->weights_list[i]->data;
        const float* b = (const float*) model->bias_list[i]->data;
        matrix_t next_result = new_matrix(batch_size, out_len, ZERO);
        // 2. batch loop (embarassingly parallelizable)
        for (unsigned int bn=0; bn<batch_size; bn++) {
            // 3. process out layer (embarassingly parallelizable)
            for (unsigned int j=0; j<out_len; j++) {
                float sum = b[j];
                // 4. process neighbours (not worth to parallelize)
                for (unsigned int k=0; k<R; k++) {
                    sum += w[j][k] * last_result->data[bn][j+k];
                }
                // since the model is a regressor, activations aren't applied to the last layer output.
                next_result->data[bn][j] = (i != last_layer) ? ACTIVATION(sum) : sum;
            }
        }
        if (i > 0) free_matrix(last_result);
        last_result = next_result;
    }
    return last_result;
}