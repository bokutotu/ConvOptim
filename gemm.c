#include <arm_neon.h>
#include <stddef.h>

#include "gemm.h"

/**
 * Naive implementation of General Matrix Multiplication (GEMM)
 * Computes C = A * B using the standard triple-nested loop approach
 * 
 * @param A Input matrix A of size m x k
 * @param B Input matrix B of size k x n
 * @param C Output matrix C of size m x n
 * @param m Number of rows in A and C
 * @param n Number of columns in B and C
 * @param k Number of columns in A and rows in B
 */
void gemm_naive(const float *A, const float *B, float *C, int m, int n, int k) {
    // Initialize result matrix C to zeros
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0.0f;
        }
    }
    
    // Perform matrix multiplication with standard triple nested loops
    for (int i = 0; i < m; i++) {         // For each row of A
        for (int j = 0; j < n; j++) {     // For each column of B
            for (int l = 0; l < k; l++) { // For each element in row/column
                C[i * n + j] += A[i * k + l] * B[l * n + j];
            }
        }
    }
}

/**
 * Tiled implementation of General Matrix Multiplication (GEMM)
 * Computes C = A * B using a 3-level tiled approach for better cache locality
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
                int M, 
                int N, 
                int K, 
                int tile_size) {
    // Initialize result matrix C to zeros
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0.0f;
        }
    }
    
    // Calculate tile sizes for each level
    int level1_tile = tile_size;           // L3 cache level
    int level2_tile = tile_size / 2;       // L2 cache level
    int level3_tile = tile_size / 4;       // L1 cache level
    
    // Ensure minimum tile sizes
    if (level2_tile < 8) level2_tile = 8;
    if (level3_tile < 4) level3_tile = 4;
    
    // Level 1 tiling (L3 cache)
    for (int i1 = 0; i1 < M; i1 += level1_tile) {
        int i1_end = (i1 + level1_tile < M) ? i1 + level1_tile : M;
        
        for (int j1 = 0; j1 < N; j1 += level1_tile) {
            int j1_end = (j1 + level1_tile < N) ? j1 + level1_tile : N;
            
            for (int k1 = 0; k1 < K; k1 += level1_tile) {
                int k1_end = (k1 + level1_tile < K) ? k1 + level1_tile : K;
                
                // Level 2 tiling (L2 cache)
                for (int i2 = i1; i2 < i1_end; i2 += level2_tile) {
                    int i2_end = (i2 + level2_tile < i1_end) ? i2 + level2_tile : i1_end;
                    
                    for (int j2 = j1; j2 < j1_end; j2 += level2_tile) {
                        int j2_end = (j2 + level2_tile < j1_end) ? j2 + level2_tile : j1_end;
                        
                        for (int k2 = k1; k2 < k1_end; k2 += level2_tile) {
                            int k2_end = (k2 + level2_tile < k1_end) ? k2 + level2_tile : k1_end;
                            
                            // Level 3 tiling (L1 cache)
                            for (int i3 = i2; i3 < i2_end; i3 += level3_tile) {
                                int i3_end = (i3 + level3_tile < i2_end) ? i3 + level3_tile : i2_end;
                                
                                for (int j3 = j2; j3 < j2_end; j3 += level3_tile) {
                                    int j3_end = (j3 + level3_tile < j2_end) ? j3 + level3_tile : j2_end;
                                    
                                    for (int k3 = k2; k3 < k2_end; k3 += level3_tile) {
                                        int k3_end = (k3 + level3_tile < k2_end) ? k3 + level3_tile : k2_end;
                                        
                                        // Actual computation at the smallest tile level
                                        for (int i = i3; i < i3_end; i++) {
                                            for (int j = j3; j < j3_end; j++) {
                                                for (int k = k3; k < k3_end; k++) {
                                                    C[i * N + j] += A[i * K + k] * B[k * N + j];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void gemm_tiled_simd(const float * restrict A, 
                     const float * restrict B, 
                     float * restrict C, 
                     int M, 
                     int N, 
                     int K, 
                     int tile_size) {
    int level1_tile = tile_size;
    int level2_tile = tile_size / 2;
    int level3_tile = tile_size / 4;
    
    if (level2_tile < 8) level2_tile = 8;
    if (level3_tile < 4) level3_tile = 4;
    
    // Level 1 (L3) タイリング
    for (int i1 = 0; i1 < M; i1 += level1_tile) {
        int i1_end = (i1 + level1_tile < M) ? i1 + level1_tile : M;
        for (int j1 = 0; j1 < N; j1 += level1_tile) {
            int j1_end = (j1 + level1_tile < N) ? j1 + level1_tile : N;
            for (int k1 = 0; k1 < K; k1 += level1_tile) {
                int k1_end = (k1 + level1_tile < K) ? k1 + level1_tile : K;
                
                // Level 2 (L2) タイリング
                for (int i2 = i1; i2 < i1_end; i2 += level2_tile) {
                    int i2_end = (i2 + level2_tile < i1_end) ? i2 + level2_tile : i1_end;
                    for (int j2 = j1; j2 < j1_end; j2 += level2_tile) {
                        int j2_end = (j2 + level2_tile < j1_end) ? j2 + level2_tile : j1_end;
                        for (int k2 = k1; k2 < k1_end; k2 += level2_tile) {
                            int k2_end = (k2 + level2_tile < k1_end) ? k2 + level2_tile : k1_end;
                            
                            // Level 3 (L1) タイリング
                            for (int i3 = i2; i3 < i2_end; i3 += level3_tile) {
                                int i3_end = (i3 + level3_tile < i2_end) ? i3 + level3_tile : i2_end;
                                for (int j3 = j2; j3 < j2_end; j3 += level3_tile) {
                                    int j3_end = (j3 + level3_tile < j2_end) ? j3 + level3_tile : j2_end;
                                    for (int k3 = k2; k3 < k2_end; k3 += level3_tile) {
                                        int k3_end = (k3 + level3_tile < k2_end) ? k3 + level3_tile : k2_end;
                                        
                                        // SIMD最適化: 内部タイルにおける演算
                                        for (int i = i3; i < i3_end; i++) {
                                            int j = j3;
                                            // 4要素ずつベクトル化して処理
                                            for (; j <= j3_end - 4; j += 4) {
                                                // C の4要素をロード
                                                float32x4_t c_vec = vld1q_f32(&C[i * N + j]);
                                                for (int k = k3; k < k3_end; k++) {
                                                    // A のスカラー値をロード
                                                    float a_val = A[i * K + k];
                                                    // B の4要素をロード
                                                    float32x4_t b_vec = vld1q_f32(&B[k * N + j]);
                                                    // c_vec += a_val * b_vec
                                                    c_vec = vmlaq_n_f32(c_vec, b_vec, a_val);
                                                }
                                                // 結果を書き戻す
                                                vst1q_f32(&C[i * N + j], c_vec);
                                            }
                                            // ベクトル化できなかった残りはスカラーで処理
                                            for (; j < j3_end; j++) {
                                                for (int k = k3; k < k3_end; k++) {
                                                    C[i * N + j] += A[i * K + k] * B[k * N + j];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#include <arm_neon.h>

// 8x8マイクロカーネル：C_block[8×8] += A_block[8×K] * B_block[K×8]
// lda, ldb, ldc はそれぞれ A, B, C の行ストライド（行幅）です。
static inline void gemm_microkernel_8x8(const float *A, const float *B, float *C,
                                          int K, int lda, int ldb, int ldc) {
    // 8行×8列分の累積レジスタを初期化（ベクトルで4要素ずつ保持）
    float32x4_t c0_0 = vdupq_n_f32(0.0f), c0_1 = vdupq_n_f32(0.0f);
    float32x4_t c1_0 = vdupq_n_f32(0.0f), c1_1 = vdupq_n_f32(0.0f);
    float32x4_t c2_0 = vdupq_n_f32(0.0f), c2_1 = vdupq_n_f32(0.0f);
    float32x4_t c3_0 = vdupq_n_f32(0.0f), c3_1 = vdupq_n_f32(0.0f);
    float32x4_t c4_0 = vdupq_n_f32(0.0f), c4_1 = vdupq_n_f32(0.0f);
    float32x4_t c5_0 = vdupq_n_f32(0.0f), c5_1 = vdupq_n_f32(0.0f);
    float32x4_t c6_0 = vdupq_n_f32(0.0f), c6_1 = vdupq_n_f32(0.0f);
    float32x4_t c7_0 = vdupq_n_f32(0.0f), c7_1 = vdupq_n_f32(0.0f);

    // Kループ：各 k に対して A の8行分と B の8列分を乗算して加算
    for (int k = 0; k < K; k++) {
        // 次回アクセスのプリフェッチ（あまり多くとりすぎないように注意）
        __builtin_prefetch(&A[(k+1) * lda + 0]);
        __builtin_prefetch(&B[(k+1) * ldb + 0]);

        // A の各行からスカラー値をロード
        float a0 = A[0 * lda + k];
        float a1 = A[1 * lda + k];
        float a2 = A[2 * lda + k];
        float a3 = A[3 * lda + k];
        float a4 = A[4 * lda + k];
        float a5 = A[5 * lda + k];
        float a6 = A[6 * lda + k];
        float a7 = A[7 * lda + k];

        // B の k 行目から、8列分を2ベクトルに分けてロード
        float32x4_t b0 = vld1q_f32(&B[k * ldb + 0]);
        float32x4_t b1 = vld1q_f32(&B[k * ldb + 4]);

        // 各行毎に乗算加算
        c0_0 = vmlaq_n_f32(c0_0, b0, a0);
        c0_1 = vmlaq_n_f32(c0_1, b1, a0);

        c1_0 = vmlaq_n_f32(c1_0, b0, a1);
        c1_1 = vmlaq_n_f32(c1_1, b1, a1);

        c2_0 = vmlaq_n_f32(c2_0, b0, a2);
        c2_1 = vmlaq_n_f32(c2_1, b1, a2);

        c3_0 = vmlaq_n_f32(c3_0, b0, a3);
        c3_1 = vmlaq_n_f32(c3_1, b1, a3);

        c4_0 = vmlaq_n_f32(c4_0, b0, a4);
        c4_1 = vmlaq_n_f32(c4_1, b1, a4);

        c5_0 = vmlaq_n_f32(c5_0, b0, a5);
        c5_1 = vmlaq_n_f32(c5_1, b1, a5);

        c6_0 = vmlaq_n_f32(c6_0, b0, a6);
        c6_1 = vmlaq_n_f32(c6_1, b1, a6);

        c7_0 = vmlaq_n_f32(c7_0, b0, a7);
        c7_1 = vmlaq_n_f32(c7_1, b1, a7);
    }

    // 結果ブロックを C に書き戻し（C は事前にゼロ初期化されているので加算）
    // ここでは、各行ごとに既存値と加算しています
    float32x4_t c0_orig0 = vld1q_f32(&C[0 * ldc + 0]);
    float32x4_t c0_orig1 = vld1q_f32(&C[0 * ldc + 4]);
    vst1q_f32(&C[0 * ldc + 0], vaddq_f32(c0_orig0, c0_0));
    vst1q_f32(&C[0 * ldc + 4], vaddq_f32(c0_orig1, c0_1));

    float32x4_t c1_orig0 = vld1q_f32(&C[1 * ldc + 0]);
    float32x4_t c1_orig1 = vld1q_f32(&C[1 * ldc + 4]);
    vst1q_f32(&C[1 * ldc + 0], vaddq_f32(c1_orig0, c1_0));
    vst1q_f32(&C[1 * ldc + 4], vaddq_f32(c1_orig1, c1_1));

    float32x4_t c2_orig0 = vld1q_f32(&C[2 * ldc + 0]);
    float32x4_t c2_orig1 = vld1q_f32(&C[2 * ldc + 4]);
    vst1q_f32(&C[2 * ldc + 0], vaddq_f32(c2_orig0, c2_0));
    vst1q_f32(&C[2 * ldc + 4], vaddq_f32(c2_orig1, c2_1));

    float32x4_t c3_orig0 = vld1q_f32(&C[3 * ldc + 0]);
    float32x4_t c3_orig1 = vld1q_f32(&C[3 * ldc + 4]);
    vst1q_f32(&C[3 * ldc + 0], vaddq_f32(c3_orig0, c3_0));
    vst1q_f32(&C[3 * ldc + 4], vaddq_f32(c3_orig1, c3_1));

    float32x4_t c4_orig0 = vld1q_f32(&C[4 * ldc + 0]);
    float32x4_t c4_orig1 = vld1q_f32(&C[4 * ldc + 4]);
    vst1q_f32(&C[4 * ldc + 0], vaddq_f32(c4_orig0, c4_0));
    vst1q_f32(&C[4 * ldc + 4], vaddq_f32(c4_orig1, c4_1));

    float32x4_t c5_orig0 = vld1q_f32(&C[5 * ldc + 0]);
    float32x4_t c5_orig1 = vld1q_f32(&C[5 * ldc + 4]);
    vst1q_f32(&C[5 * ldc + 0], vaddq_f32(c5_orig0, c5_0));
    vst1q_f32(&C[5 * ldc + 4], vaddq_f32(c5_orig1, c5_1));

    float32x4_t c6_orig0 = vld1q_f32(&C[6 * ldc + 0]);
    float32x4_t c6_orig1 = vld1q_f32(&C[6 * ldc + 4]);
    vst1q_f32(&C[6 * ldc + 0], vaddq_f32(c6_orig0, c6_0));
    vst1q_f32(&C[6 * ldc + 4], vaddq_f32(c6_orig1, c6_1));

    float32x4_t c7_orig0 = vld1q_f32(&C[7 * ldc + 0]);
    float32x4_t c7_orig1 = vld1q_f32(&C[7 * ldc + 4]);
    vst1q_f32(&C[7 * ldc + 0], vaddq_f32(c7_orig0, c7_0));
    vst1q_f32(&C[7 * ldc + 4], vaddq_f32(c7_orig1, c7_1));
}

// 3段階タイル化＋プリフェッチおよび8x8マイクロカーネルを用いた GEMM 実装
void gemm_tiled_simd_prefetch(const float * restrict A, 
                              const float * restrict B, 
                              float * restrict C, 
                              int M, int N, int K, 
                              int tile_size) {
    // タイルサイズの設定
    int level1_tile = tile_size;         // L3キャッシュレベル用
    int level2_tile = tile_size / 2;       // L2キャッシュレベル用
    // マイクロカーネルは8x8ブロックを前提とするので、最低レベルは8以上に設定
    int level3_tile = (tile_size / 4 < 8) ? 8 : tile_size / 4;
    
    // Level 1 タイリング (L3キャッシュ)
    for (int i1 = 0; i1 < M; i1 += level1_tile) {
        int i1_end = (i1 + level1_tile < M) ? i1 + level1_tile : M;
        for (int j1 = 0; j1 < N; j1 += level1_tile) {
            int j1_end = (j1 + level1_tile < N) ? j1 + level1_tile : N;
            for (int k1 = 0; k1 < K; k1 += level1_tile) {
                int k1_end = (k1 + level1_tile < K) ? k1 + level1_tile : K;
                
                // Level 2 タイリング (L2キャッシュ)
                for (int i2 = i1; i2 < i1_end; i2 += level2_tile) {
                    int i2_end = (i2 + level2_tile < i1_end) ? i2 + level2_tile : i1_end;
                    for (int j2 = j1; j2 < j1_end; j2 += level2_tile) {
                        int j2_end = (j2 + level2_tile < j1_end) ? j2 + level2_tile : j1_end;
                        for (int k2 = k1; k2 < k1_end; k2 += level2_tile) {
                            int k2_end = (k2 + level2_tile < k1_end) ? k2 + level2_tile : k1_end;
                            
                            // Level 3 タイリング (L1キャッシュ)－ここで8x8ブロック単位で処理
                            for (int i3 = i2; i3 < i2_end; i3 += level3_tile) {
                                int i3_end = (i3 + level3_tile < i2_end) ? i3 + level3_tile : i2_end;
                                for (int j3 = j2; j3 < j2_end; j3 += level3_tile) {
                                    int j3_end = (j3 + level3_tile < j2_end) ? j3 + level3_tile : j2_end;
                                    for (int k3 = k2; k3 < k2_end; k3 += level3_tile) {
                                        int k3_end = (k3 + level3_tile < k2_end) ? k3 + level3_tile : k2_end;
                                        
                                        // マイクロカーネルによる8x8ブロック処理（ブロックサイズが不足している場合はフォールバック）
                                        for (int i = i3; i < i3_end; i += 8) {
                                            for (int j = j3; j < j3_end; j += 8) {
                                                int current_block_rows = ((i + 8) <= i3_end) ? 8 : (i3_end - i);
                                                int current_block_cols = ((j + 8) <= j3_end) ? 8 : (j3_end - j);
                                                if (current_block_rows == 8 && current_block_cols == 8) {
                                                    // サブブロックへのポインタ設定
                                                    const float *A_block = &A[i * K + k3];
                                                    const float *B_block = &B[k3 * N + j];
                                                    float *C_block = &C[i * N + j];
                                                    int current_K = k3_end - k3;
                                                    // マイクロカーネル実行前にプリフェッチ
                                                    __builtin_prefetch(A_block + 16);
                                                    __builtin_prefetch(B_block + 16);
                                                    gemm_microkernel_8x8(A_block, B_block, C_block, current_K,
                                                                          K, N, N);
                                                } else {
                                                    // ブロックサイズが8未満の場合は従来のスカラーコードで処理
                                                    for (int ii = i; ii < i + current_block_rows; ii++) {
                                                        for (int jj = j; jj < j + current_block_cols; jj++) {
                                                            for (int kk = k3; kk < k3_end; kk++) {
                                                                C[ii * N + jj] += A[ii * K + kk] * B[kk * N + jj];
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

