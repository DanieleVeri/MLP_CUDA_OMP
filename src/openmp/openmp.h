/**
 * @file openmp.h
 * @author Daniele Verì
 * @brief OpenMP implementation.
 */

#ifndef __OPENMP__H
#define __OPENMP__H

#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#include "../constants.h"
#include "../utils/model.h"

/**
 * Computes the MLP regressor output parallelizing with OpenMP.
 * @param matrix_t input_batch where columns are features and rows are batch elements.
 * @param model_t model which contains the list of weights and biases.
 * @return matrix_t containing the output.
 * */
matrix_t omp_forward_mlp(matrix_t input_batch, model_t model);

#endif