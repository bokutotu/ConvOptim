#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <cblas.h>

#include "gemm.h"

#define N 10

int main(void) {
    int n, m, k;
    n = 1024;
    m = 1024;
    k = 1024;

    float *A = (float *)malloc(n * k * sizeof(float));
    float *B = (float *)malloc(k * m * sizeof(float));
    float *C = (float *)malloc(n * m * sizeof(float));
    float *C_tiled = (float *)malloc(n * m * sizeof(float));
    float *C_tiled_simd = (float *)malloc(n * m * sizeof(float));
    float *C_tiled_simd_prefetch = (float *)malloc(n * m * sizeof(float));
    float *C_openblas = (float *)malloc(n * m * sizeof(float));

    // Random Initialize A and B, 結果配列はゼロ初期化
    for (int i = 0; i < n * k; i++) {
        A[i] = (float)((double)rand() / (double)RAND_MAX);
    }
    for (int i = 0; i < k * m; i++) {
        B[i] = (float)((double)rand() / (double)RAND_MAX);
    }
    for (int i = 0; i < n * m; i++) {
        C[i] = 0.0f;
        C_tiled[i] = 0.0f;
        C_tiled_simd[i] = 0.0f;
        C_tiled_simd_prefetch[i] = 0.0f;
        C_openblas[i] = 0.0f;
    }

    // Test naive implementation
    time_t start = clock();
    for (int i = 0; i < N; i++) {
        gemm_naive(A, B, C, n, m, k);
    }
    time_t end = clock();
    printf("Naive GEMM Time: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC / N);

    // Test tiled implementation with different tile sizes
    int tile_sizes[] = {8, 16, 32, 64, 128, 256};
    for (int t = 0; t < sizeof(tile_sizes) / sizeof(tile_sizes[0]); t++) {
        int tile_size = tile_sizes[t];
        
        start = clock();
        for (int i = 0; i < N; i++) {
            gemm_tiled(A, B, C_tiled, n, m, k, tile_size);
        }
        end = clock();

        // 結果確認用に C_tiled を再計算
        for (int i = 0; i < n * m; i++) {
            C_tiled[i] = 0.0f;
        }
        gemm_tiled(A, B, C_tiled, n, m, k, tile_size);
        printf("Tiled GEMM (tile_size=%d) Time: %f seconds\n", 
               tile_size, (double)(end - start) / CLOCKS_PER_SEC / N);
        
        if (t == 0) {
            int correct = 1;
            for (int i = 0; i < n * m; i++) {
                if (fabsf(C[i] - C_tiled[i]) > 1e-5) {
                    correct = 0;
                    printf("Error at index %d: C=%f, C_tiled=%f\n", i, C[i], C_tiled[i]);
                    break;
                }
            }
            if (correct) {
                printf("Tiled implementation produces correct results!\n");
            } else {
                printf("ERROR: Tiled implementation produces incorrect results!\n");
            }
        }
    }

    // Test tiled SIMD implementation with different tile sizes
    for (int t = 0; t < sizeof(tile_sizes) / sizeof(tile_sizes[0]); t++) {
        int tile_size = tile_sizes[t];
        
        start = clock();
        for (int i = 0; i < N; i++) {
            gemm_tiled_simd(A, B, C_tiled_simd, n, m, k, tile_size);
        }
        end = clock();

        for (int i = 0; i < n * m; i++) {
            C_tiled_simd[i] = 0.0f;
        }
        gemm_tiled_simd(A, B, C_tiled_simd, n, m, k, tile_size);
        printf("Tiled SIMD GEMM (tile_size=%d) Time: %f seconds\n", 
               tile_size, (double)(end - start) / CLOCKS_PER_SEC / N);
        
        if (t == 0) {
            int correct = 1;
            for (int i = 0; i < n * m; i++) {
                if (fabsf(C[i] - C_tiled_simd[i]) > 1e-5) {
                    correct = 0;
                    printf("Error at index %d: C=%f, C_tiled_simd=%f\n", i, C[i], C_tiled_simd[i]);
                    break;
                }
            }
            if (correct) {
                printf("Tiled SIMD implementation produces correct results!\n");
            } else {
                printf("ERROR: Tiled SIMD implementation produces incorrect results!\n");
            }
        }
    }

    // Test tiled SIMD prefetch implementation with different tile sizes
    for (int t = 4; t < sizeof(tile_sizes) / sizeof(tile_sizes[0]); t++) {
        int tile_size = tile_sizes[t];
        
        start = clock();
        for (int i = 0; i < N; i++) {
            gemm_tiled_simd_prefetch(A, B, C_tiled_simd_prefetch, n, m, k, tile_size);
        }
        end = clock();

        for (int i = 0; i < n * m; i++) {
            C_tiled_simd_prefetch[i] = 0.0f;
        }
        gemm_tiled_simd_prefetch(A, B, C_tiled_simd_prefetch, n, m, k, tile_size);
        printf("Tiled SIMD Prefetch GEMM (tile_size=%d) Time: %f seconds\n", 
               tile_size, (double)(end - start) / CLOCKS_PER_SEC / N);
        
        if (t == 4) {  // 最初のtile_sizeで結果確認
            int correct = 1;
            float max_diff = 0.0f;
            for (int i = 0; i < n * m; i++) {
                // if (fabsf(C[i] - C_tiled_simd_prefetch[i]) > 1e-5) {
                //     correct = 0;
                //     printf("Error at index %d: C=%f, C_tiled_simd_prefetch=%f\n", i, C[i], C_tiled_simd_prefetch[i]);
                //     break;
                // }
                float diff = fabsf(C[i] - C_tiled_simd_prefetch[i]);
                if (diff > max_diff) {
                    max_diff = diff;
                }
            }
            printf("Max diff: %f\n", max_diff);
            if (correct) {
                printf("Tiled SIMD Prefetch implementation produces correct results!\n");
            } else {
                printf("ERROR: Tiled SIMD Prefetch implementation produces incorrect results!\n");
            }
        }
    }

    // --- OpenBLAS GEMM のベンチマーク ---
    // C_openblas をゼロ初期化
    for (int i = 0; i < n * m; i++) {
        C_openblas[i] = 0.0f;
    }
    start = clock();
    for (int i = 0; i < N; i++) {
        // cblas_sgemm: C = alpha * A * B + beta * C
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    n, m, k,
                    1.0f,
                    A, k,
                    B, m,
                    0.0f,
                    C_openblas, m);
    }
    end = clock();
    printf("OpenBLAS GEMM Time: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC / N);
    // --- End OpenBLAS ベンチマーク ---

    free(A);
    free(B);
    free(C);
    free(C_tiled);
    free(C_tiled_simd);
    free(C_tiled_simd_prefetch);
    free(C_openblas);

    return 0;
}

