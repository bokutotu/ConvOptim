# Compiler and flags
CC = /opt/homebrew/opt/llvm/bin/clang
CFLAGS = -Wall -Wextra -O3 -std=c99 -I/opt/homebrew/Cellar/openblas/0.3.29/include -funroll-loops -march=native -flto

# OpenBLAS libraries
LIBS = -L/opt/homebrew/Cellar/openblas/0.3.29/lib -lopenblas

# Source files and target
SRCS = main.c naive.c im2col.c gemm.c
OBJS = $(SRCS:.c=.o)
TARGET = convolution

# Header files for dependency tracking
HDRS = conv.h gemm.h

# Main target
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@ $(LIBS)

# Pattern rule for object files
%.o: %.c $(HDRS)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean target
clean:
	rm -f $(OBJS) $(TARGET)

# Phony targets
.PHONY: all clean

