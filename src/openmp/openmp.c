#include "openmp.h"

matrix_t omp_forward_mlp(matrix_t input_batch, model_t model)
{
    const unsigned int batch_size = input_batch->m;
    const unsigned int last_layer = model->num_layer-1;
    // buffer holding the last layer output
    matrix_t last_result = input_batch;
    // layer loop
    for (unsigned int i=0; i<model->num_layer; i++) {
        const unsigned int out_len = model->bias_list[i]->len;
        const float** w = (const float**) model->weights_list[i]->data;
        const float* b = (const float*) model->bias_list[i]->data;
        matrix_t next_result = new_matrix(batch_size, out_len, ZERO);

        // omp directives: parallelize over both the elements of the layer output and the batch elements; 
        // schedule type static since the predictale amount of work; 
        //default scope set to none as best practice, used variable are firstprivate. 
        #pragma omp parallel for collapse(2) \
            schedule(static) \
            default(none) firstprivate(w, b, next_result, last_result, i) 
        for (unsigned int bn=0; bn<batch_size; bn++) {
            for (unsigned int j=0; j<out_len; j++) {
                float sum = b[j];
                // since R is small by definition no need to parallelize (too overhead).
                for (unsigned int k=0; k<R; k++) {
                    sum += w[j][k] * last_result->data[bn][j+k];
                }
                // since the model is a regressor, activations aren't applied to the last layer output.
                next_result->data[bn][j] = (i != last_layer) ? ACTIVATION(sum) : sum;
            }
        }
        // move next_result in last_result
        if (i > 0) free_matrix(last_result);
        last_result = next_result;
    }
    return last_result;
}