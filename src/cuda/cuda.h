#ifndef __CUDA__H
#define __CUDA__H

#ifdef __cplusplus
extern "C" {
#endif

#include "../utils/log.h" //todo: remove
#include "../constants.h"
#include "../utils/model.h"

matrix_t cuda1_forward_mlp(matrix_t input_batch, model_t model);

vector_t h2d_vector(vector_t vec);
vector_t d2h_vector(vector_t d_vec);
void device_free_vector(vector_t d_vec);

matrix_t h2d_matrix(matrix_t mat);
matrix_t d2h_matrix(matrix_t d_mat);
void device_free_matrix(matrix_t d_mat);

#ifdef __cplusplus
}
#endif

#endif