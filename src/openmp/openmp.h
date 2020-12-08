#ifndef __OPENMP__H
#define __OPENMP__H

#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#include "../constants.h"
#include "../utils/model.h"

vector_t serial_forward_mlp(vector_t input, model_t model);

#endif