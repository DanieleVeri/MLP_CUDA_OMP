.PHONY: clean openmp cuda dist
CC = gcc 
CFLAGS = -std=c99 -Wall -Wpedantic -g
OMP_CFLAGS = -fopenmp 
OMP_LIBS = -lgomp -lm
OMP_INCLUDES = -Isrc -Isrc/utils -Isrc/openmp
OMP_SRCS = src/main_omp.c $(wildcard src/utils/*.c) $(wildcard src/openmp/*.c)
OMP_MAIN = mlp_omp

CUDA_CC = nvcc
CUDA_FLAGS =
cuda_legacy: CUDA_FLAGS += --gpu-architecture compute_20 --Wno-deprecated-gpu-targets
CUDA_INCLUDES = -Isrc -Isrc/utils -Isrc/cuda
CUDA_SRCS = $(wildcard src/cuda/*.cu)
CUDA_C_SRCS = src/main_cuda.c $(wildcard src/utils/*.c)
CUDA_MAIN = mlp_cuda

# build targets
all:	openmp cuda
	@echo --All targets are built.
openmp:	$(OMP_MAIN)
	@echo --Builded target OPENMP
cuda cuda_legacy: $(CUDA_MAIN)
	@echo --Builded target CUDA
clean:
	$(RM) src/*.o src/**/*.o *~ $(CUDA_MAIN) $(OMP_MAIN)

# run targets
run_omp_4: openmp
	OMP_NUM_THREADS=4 ./$(OMP_MAIN) 20001 5000
run_cuda: cuda
	./$(CUDA_MAIN)

dist: clean
	@zip -r VeriDaniele.zip .
	@echo --Generated zip

OMP_OBJS = $(OMP_SRCS:.c=.o)
CUDA_OBJS = $(CUDA_SRCS:.cu=.o)
CUDA_C_OBJS = $(CUDA_C_SRCS:.c=_c_.o)

$(OMP_OBJS): %.o: %.c
	$(CC) $(CFLAGS) $(OMP_CFLAGS) $(OMP_INCLUDES) -c $< -o $@
$(CUDA_OBJS): %.o: %.cu
	$(CUDA_CC) $(CUDA_FLAGS) $(CUDA_INCLUDES) -c $< -o $@
$(CUDA_C_OBJS): %_c_.o: %.c
	$(CC) $(CFLAGS) $(CUDA_INCLUDES) -c $< -o $@

$(OMP_MAIN): $(OMP_OBJS)
	$(CC) $(CFLAGS) $(OMP_CFLAGS) $(OMP_INCLUDES) -o $(OMP_MAIN) $(OMP_OBJS) $(OMP_LIBS)
$(CUDA_MAIN): $(CUDA_OBJS) $(CUDA_C_OBJS)
	$(CUDA_CC) $(CUDA_FLAGS) $(CUDA_INCLUDES) -o $(CUDA_MAIN) $(CUDA_OBJS) $(CUDA_C_OBJS)