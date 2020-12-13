#ifndef __OPENMP__H
#define __OPENMP__H

#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#include "../constants.h"
#include "../utils/model.h"

matrix_t omp1_forward_mlp(matrix_t input_batch, model_t model);

#endif