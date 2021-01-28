/**
 * @file cuda.h
 * @author Daniele Verì
 * @brief CUDA implementation.
 */

#ifndef __CUDA__H
#define __CUDA__H

#ifdef __cplusplus
extern "C" {
#endif

#include "../constants.h"
#include "../utils/model.h"

/**
 * Computes the MLP regressor output parallelizing with CUDA.
 * @param matrix_t input_batch where columns are features and rows are batch elements.
 * @param model_t model which contains the list of weights and biases.
 * @return matrix_t containing the output.
 * */
matrix_t cuda_forward_mlp(matrix_t input_batch, model_t model);

/********************************** VECTOR MEMORY UTILITIES **********************************/

/**
 * Allocates a memory pinned float vector with the given length and initialize it.
 * @param int length of the vector
 * @param init_t init type
 * @return vector_t
 */
vector_t new_vector_pinned(unsigned int length, init_t init_type);

/**
 * Deallocates the mem pinned vector.
 * @param vector_t vector to free
 */
void free_vector_pinned(vector_t h_vec);

/**
 * Copy vector from host memory to device memory.
 * @param vector_t host pointer
 * @return vector_t device pointer
 * */
vector_t h2d_vector(vector_t vec);

/**
 * Copy vector from device memory to host memory.
 * @param vector_t device pointer
 * @return vector_t host pointer
 * */
vector_t d2h_vector(vector_t d_vec);

/**
 * Free vector on device memory.
 * @param vector_t device pointer
 * */
void device_free_vector(vector_t d_vec);

/********************************** MATRIX MEMORY UTILITIES **********************************/

/**
 * Allocates a memory pinned float matrix with the given rows and cols and initialize it.
 * Data are allocated in a contiguous memory region.
 * @param int rows
 * @param int cols
 * @param init_t init type
 * @return matrix_t
 */
matrix_t new_matrix_pinned(unsigned int m, unsigned int n, init_t init_type);

/**
 * Deallocates the mem pinned matrix.
 * @param matrix_t matrix to free
 */
void free_matrix_pinned(matrix_t h_mat);

/**
 * Copy matrix from host memory to device memory.
 * @param matrix_t host pointer
 * @return matrix_t device pointer
 * */
matrix_t h2d_matrix(matrix_t mat);

/**
 * Copy matrix from device memory to host memory.
 * @param matrix_t device pointer
 * @return matrix_t host pointer
 * */
matrix_t d2h_matrix(matrix_t d_mat);

/**
 * Free matrix on device memory.
 * @param matrix_t device pointer
 * */
void device_free_matrix(matrix_t d_mat);

/********************************** MODEL MEMORY UTILITIES **********************************/

/**
 * Allocates a memory pinned list of weights (matrices) and biases (vectors) 
 * that make the model.
 * @param int nom of input features
 * @param int num layers
 * @param init_t init type
 * @return model_t
 */
model_t new_model_pinned(unsigned int inputs, unsigned int num_layer, init_t init_type);

/**
 * Deallocates the mem pinned model.
 * @param model_t
 */
void free_model_pinned(model_t h_model);

/**
 * Copy model from host to device memory.
 * @param model_t host pointer
 * @return model_t device pointer
 * */
model_t h2d_model(model_t mdl);

/**
 * Free model on device memory.
 * @param model_t device pointer
 * */
void device_free_model(model_t d_mdl);

/**
 * Performs numerous allocations, host to device transfers,
 * device to host transfers and deallocations in order
 * to identify easily memory leaks.
 * */
void test_device_mem_leak();

#ifdef __cplusplus
}
#endif

#endif