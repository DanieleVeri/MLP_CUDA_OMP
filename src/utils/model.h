#ifndef __MODEL__H
#define __MODEL__H

#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "../constants.h"

// threshold used for comparisons
#define EPS 1e-5

// activations
#define RELU(x) ((x) * ((x) > 0))       // branchless relu
#define SIGM(x) (1/(1+exp(-(x))))       // sigmoid
#define ACTIVATION SIGM                 // activation selection

// initialization types of vector and matrices (random range: [-2, +2))
typedef enum {ZERO, RAND_UNIFORM} init_t;

/************************* vector data type *************************/
typedef struct {
    unsigned int len;
    float* data;
} vector_s;
typedef vector_s* vector_t;

/**
 * Allocates a float vector with the given length and initialize it.
 * @param int length of the vector
 * @param init_t init type
 * @return vector_t
 */
vector_t new_vector(unsigned int length, init_t init_type);

/**
 * Deallocates the vector.
 * @param vector_t vector to free
 */
void free_vector(vector_t vector);

/**
 * Assert the equality between two vectors.
 * Elementwise difference must be < EPS.
 * @param vector_t v1
 * @param vector_t v2
 */
void assert_equal_vector(vector_t v1, vector_t v2);

/************************* matrix data type *************************/
typedef struct {
    unsigned int m;
    unsigned int n;
    float** data;
} matrix_s;
typedef matrix_s* matrix_t;

/**
 * Allocates a float matrix with the given rows and cols and initialize it.
 * Data are allocated in a contiguous memory region.
 * @param int rows
 * @param int cols
 * @param init_t init type
 * @return matrix_t
 */
matrix_t new_matrix(unsigned int m, unsigned int n, init_t init_type);

/**
 * Deallocates the matrix.
 * @param matrix_t matrix to free
 */
void free_matrix(matrix_t matrix);

/**
 * Assert the equality between two matrices.
 * Elementwise difference must be < EPS.
 * @param matrix_t m1
 * @param matrix_t m2
 */
void assert_equal_matrix(matrix_t m1, matrix_t m2);

/************************* model data type *************************/
typedef struct {
    unsigned int num_layer;
    matrix_t* weights_list;
    vector_t* bias_list;
} model_s;
typedef model_s* model_t;

/**
 * Allocates a list of weights (matrices) and biases (vectors) 
 * that make the model.
 * @param int nom of input features
 * @param int num layers
 * @param init_t init type
 * @return model_t
 */
model_t new_model(unsigned int inputs, unsigned int num_layer, init_t init_type);

/**
 * Deallocates the model.
 * @param model_t
 */
void free_model(model_t layers);

/**
 * Serial implementation of the MLP regressor.
 * @param matrix_t input_batch where columns are features and rows are batch elements.
 * @param model_t model which contains the list of weights and biases.
 * @return matrix_t containing the output.
 */
matrix_t serial_forward_mlp(matrix_t input_batch, model_t model);

#endif