.PHONY: clean openmp cuda dist

COMMON_INCLUDES = -Isrc -Isrc/utils
COMMON_SOURCES = $(wildcard src/utils/*.c)

OMP_CC = gcc
OMP_CFLAGS = -Wall -g -fopenmp 
OMP_LIBS = -lgomp
OMP_INCLUDES = $(COMMON_INCLUDES) -Isrc/openmp
OMP_SRCS = $(COMMON_SOURCES) src/main_omp.c $(wildcard src/openmp/*.c)
OMP_MAIN = mlp_omp

CUDA_CC = nvcc
CUDA_INCLUDES = $(COMMON_INCLUDES) -Isrc/cuda
CUDA_SRCS = $(COMMON_SOURCES) src/main_cuda.cu $(wildcard src/cuda/*.cu)
CUDA_MAIN = mlp_cuda

all:	openmp cuda
	@echo --All targets are builded
openmp:	$(OMP_MAIN)
	@echo --Builded target OPENMP
cuda:	$(CUDA_MAIN)
	@echo --Builded target CUDA
clean:
	$(RM) src/*.o src/**/*.o *~ $(CUDA_MAIN) $(OMP_MAIN)
dist: clean
	@zip -r VeriDaniele.zip .
	@echo --Generated zip

OMP_OBJS = $(OMP_SRCS:.c=.o)
CUDA_OBJS = $(CUDA_SRCS:.c=.o)

$(OMP_MAIN): $(OMP_OBJS)
	$(OMP_CC) $(OMP_CFLAGS) $(OMP_INCLUDES) -o $(OMP_MAIN) $(OMP_OBJS) $(OMP_LIBS)
$(CUDA_MAIN): $(CUDA_OBJS)
	$(CUDA_CC) $(CUDA_INCLUDES) -o $(CUDA_MAIN) $(CUDA_OBJS)

OMP_OBJS:	$(OMP_SRCS)
	$(OMP_CC) $(OMP_CFLAGS) $(OMP_INCLUDES) -c $<  -o $@
CUDA_OBJS:	$(CUDA_SRCS)
	$(CUDA_CC) $(CUDA_INCLUDES) -c $<  -o $@