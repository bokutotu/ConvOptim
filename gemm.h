#ifndef GEMM_H
#define GEMM_H

/**
 * Naive implementation of General Matrix Multiplication (GEMM)
 * Computes C = A * B
 * 
 * @param A Input matrix A of size m x k
 * @param B Input matrix B of size k x n
 * @param C Output matrix C of size m x n
 * @param m Number of rows in A and C
 * @param n Number of columns in B and C
 * @param k Number of columns in A and rows in B
 */
void gemm_naive(const float *A, const float *B, float *C, int m, int n, int k);

/**
 * Tiled implementation of General Matrix Multiplication (GEMM)
 * Computes C = A * B using a tiled approach for better cache locality
 * 
 * @param A Input matrix A of size m x k
 * @param B Input matrix B of size k x n
 * @param C Output matrix C of size m x n
 * @param m Number of rows in A and C
 * @param n Number of columns in B and C
 * @param k Number of columns in A and rows in B
 * @param tile_size Size of tiles for tiling optimization
 */
void gemm_tiled(const float * restrict A, 
                const float * restrict B, 
                float * restrict C, 
                int m, 
                int n, 
                int k, 
                int tile_size);


/**
 * Tiled implementation and SIMD of General Matrix Multiplication (GEMM)
 * Computes C = A * B using a tiled approach for better cache locality
 * 
 * @param A Input matrix A of size m x k
 * @param B Input matrix B of size k x n
 * @param C Output matrix C of size m x n
 * @param m Number of rows in A and C
 * @param n Number of columns in B and C
 * @param k Number of columns in A and rows in B
 * @param tile_size Size of tiles for tiling optimization
 */
void gemm_tiled_simd(const float * restrict A, 
                     const float * restrict B, 
                     float * restrict C, 
                     int m, 
                     int n, 
                     int k, 
                     int tile_size);

/**
 * Tiled implementation and SIMD MicroKernel of General Matrix Multiplication (GEMM)
 * Computes C = A * B using a tiled approach for better cache locality
 * 
 * @param A Input matrix A of size m x k
 * @param B Input matrix B of size k x n
 * @param C Output matrix C of size m x n
 * @param m Number of rows in A and C
 * @param n Number of columns in B and C
 * @param k Number of columns in A and rows in B
 * @param tile_size Size of tiles for tiling optimization
 */
void gemm_tiled_simd_prefetch(const float * restrict A, 
                              const float * restrict B, 
                              float * restrict C, 
                              int m, 
                              int n, 
                              int k, 
                              int tile_size);

#endif /* GEMM_H */
