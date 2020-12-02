DISTNAME = VeriDaniele
CC = gcc
CFLAGS = -Wall -g -fopenmp 
#LFLAGS = 
COMMON_INCLUDES = -Isrc
COMMON_SOURCES = src/main.c

MP_INCLUDES = $(COMMON_INCLUDES) -Isrc/openmp
CUDA_INCLUDES = $(COMMON_INCLUDES) -Isrc/cuda

MP_SRCS = $(COMMON_SOURCES) src/openmp/openmp.c
CUDA_SRCS = $(COMMON_SOURCES) src/cuda/cuda.c 

MP_LIBS = -lgomp
CUDA_LIBS = -lgomp

MP_MAIN = mlp_mp
CUDA_MAIN = mlp_cuda

.PHONY: clean openmp cuda dist
all:	openmp cuda
openmp:	$(MP_MAIN)
	@echo --Builded target OPENMP
cuda:	$(CUDA_MAIN)
	@echo --Builded target CUDA
clean:
	$(RM) src/*.o src/**/*.o *~ $(CUDA_MAIN) $(MP_MAIN)
dist: clean
	@zip -r $(DISTNAME).zip .
	@echo --Generated zip

MP_OBJS = $(MP_SRCS:.c=.o)
CUDA_OBJS = $(CUDA_SRCS:.c=.o)

$(MP_MAIN): $(MP_OBJS)
	$(CC) $(CFLAGS) $(MP_INCLUDES) -o $(MP_MAIN) $(MP_OBJS) $(LFLAGS) $(MP_LIBS)
$(CUDA_MAIN): $(CUDA_OBJS)
	$(CC) $(CFLAGS) $(CUDA_INCLUDES) -o $(CUDA_MAIN) $(CUDA_OBJS) $(LFLAGS) $(CUDA_LIBS)

MP_OBJS:	$(MP_SRCS)
	$(CC) $(CFLAGS) $(MP_INCLUDES) -c $<  -o $@
CUDA_OBJS:	$(CUDA_SRCS)
	$(CC) $(CFLAGS) $(CUDA_INCLUDES) -c $<  -o $@