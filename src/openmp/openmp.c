#include "openmp.h"

vector_t serial_forward_mlp(vector_t input, model_t model)
{
    #pragma omp parallel
    {
        int my_rank = omp_get_thread_num();
        int thread_count = omp_get_num_threads();
        printf("Hello from thread %d of %d\n", my_rank, thread_count);
    }

    for (unsigned int i=0; i<model.num_layer-1; i++) {
        const fp** w = (const fp**) model.weights_list[i].data;
        const fp* b = (const fp*) model.bias_list[i].data;
        const unsigned int out_len = model.bias_list[i].len;
        const fp* x = input.data;

        //for (unsigned int b=0; b<batch_len; b++)
        vector_t layer_out = new_vector(out_len, ZERO);
        for (unsigned int j=0; j<out_len; j++) {
            fp sum = b[j];
            for (unsigned int k=0; k<R; k++) {
                sum += w[j][k] * x[j+k];
            }
            layer_out.data[j] = ACTIVATION(sum);
        }

        if (i > 0) free_vector(input);
        input = layer_out;
    }

    // Last layer
    const unsigned int last_layer_idx = model.num_layer-1;
    const fp** w = (const fp**) model.weights_list[last_layer_idx].data;
    const fp* b = (const fp*) model.bias_list[last_layer_idx].data;
    const unsigned int out_len = model.bias_list[last_layer_idx].len;
    const fp* x = input.data;

    //for (unsigned int b=0; b<batch_len; b++)
    vector_t layer_out = new_vector(out_len, ZERO);
    for (unsigned int j=0; j<out_len; j++) {
        fp sum = b[j];
        for (unsigned int k=0; k<R; k++) {
            sum += w[j][k] * x[j+k];
        }
        layer_out.data[j] = ID(sum); //softmax
    }
    free_vector(input);

    return layer_out;
}
