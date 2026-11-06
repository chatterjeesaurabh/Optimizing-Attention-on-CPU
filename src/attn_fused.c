/*
 * attn_fused.c -- Q3c: row-wise fusion (the N x N matrix is never stored).
 *
 * Placeholder: replace the body of attention() with your implementation.
 * Keep the driver (main) below unchanged. See attn_naive.c for the input and
 * output format.
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------------- */
/* attention(): write your implementation here.                               */
/* ------------------------------------------------------------------------- */
/*
 * Row-wise fusion: for each strip of MR query rows we compute the scores
 * against all N keys, do the softmax and multiply by V before moving on, so
 * only an MR x N strip of S exists at any time (MR = 6: 6 x 4096 x 4 B =
 * 96 KB, which lives in L2) instead of the 64 MB N x N matrix that
 * attn_tiled.c writes to and re-reads from DRAM. The strip is the
 * granularity of the register tile; with -DMR=1 this is literally row by
 * row (one 16 KB row in L1) but every K/V vector loaded is then used by a
 * single FMA.
 * The QK^T and PV micro-kernels are the register-tiled ones of attn_tiled.c;
 * the PV tile now accumulates over all N keys in registers. The softmax
 * normalisation is applied to the d outputs instead of the N probabilities.
 */
#ifndef MR
#define MR 6   /* query rows processed together (register tile height) */
#endif
#define NR 32  /* keys per register tile in S = Q K^T         */
#define DN 32  /* output columns per register tile in O = P V */

static size_t round_up(size_t x, size_t m) { return (x + m - 1) / m * m; }

/* S[0..MR)[0..NR) = Qp[0..MR)[0..d) * Kp (d x NR panel) */
static inline void qk_tile(const float *restrict Qp, const float *restrict Kp, float *restrict S,
                           size_t ldS, int d)
{
    float acc[MR][NR] = {{0}};
    for (int k = 0; k < d; k++)
        for (int r = 0; r < MR; r++) {
            const float q = Qp[(size_t)r * d + k];
            for (int c = 0; c < NR; c++)
                acc[r][c] += q * Kp[(size_t)k * NR + c];
        }
    for (int r = 0; r < MR; r++)
        for (int c = 0; c < NR; c++)
            S[r * ldS + c] = acc[r][c];
}

/* Ot[0..MR)[0..DN) = (P[0..MR)[0..nj) * V[0..nj)[0..DN)) * inv[r] */
static inline void pv_tile(const float *restrict P, size_t ldP, const float *restrict V, size_t ldV,
                           float *restrict Ot, size_t ldO, int nj, const float *restrict inv)
{
    float acc[MR][DN] = {{0}};
    for (int j = 0; j < nj; j++)
        for (int r = 0; r < MR; r++) {
            const float p = P[r * ldP + j];
            for (int c = 0; c < DN; c++)
                acc[r][c] += p * V[(size_t)j * ldV + c];
        }
    for (int r = 0; r < MR; r++)
        for (int c = 0; c < DN; c++)
            Ot[r * ldO + c] = acc[r][c] * inv[r];
}

void attention(const float *restrict Q, const float *restrict K, const float *restrict V,
               float *restrict O, int N, int d)
{
    const float scale = 1.0f / sqrtf((float)d);
    const size_t Np = round_up(N, MR), Nc = round_up(N, NR), dp = round_up(d, DN);

    float *Qp = calloc(Np * d, sizeof(float));  /* Q * scale, rows padded */
    float *Kp = calloc(Nc * d, sizeof(float));  /* K^T in NR-wide panels  */
    float *Vp = calloc((size_t)N * dp, sizeof(float));
    float *Op = calloc(Np * dp, sizeof(float));
    float *S = malloc(MR * Nc * sizeof(float)); /* one MR-row strip of S  */
    float inv[MR];

    for (size_t i = 0; i < (size_t)N * d; i++)
        Qp[i] = Q[i] * scale;
    for (int j = 0; j < N; j++)
        for (int k = 0; k < d; k++)
            Kp[(size_t)(j / NR) * NR * d + (size_t)k * NR + j % NR] = K[(size_t)j * d + k];
    for (int j = 0; j < N; j++)
        memcpy(Vp + (size_t)j * dp, V + (size_t)j * d, d * sizeof(float));

    for (size_t i = 0; i < Np; i += MR) {
        /* 1. scores of rows i..i+MR against all keys */
        for (size_t j = 0; j < Nc; j += NR)
            qk_tile(Qp + i * d, Kp + j * d, S + j, Nc, d);

        /* 2. unnormalised softmax of each row of the strip */
        for (int r = 0; r < MR; r++) {
            float *restrict row = S + (size_t)r * Nc;
            float max = row[0];
            for (int j = 1; j < N; j++)
                max = row[j] > max ? row[j] : max;
            float sum = 0.0f;
            for (int j = 0; j < N; j++) {
                row[j] = expf(row[j] - max);
                sum += row[j];
            }
            inv[r] = 1.0f / sum;
        }

        /* 3. O[i..i+MR) = P V / sum */
        for (size_t c = 0; c < dp; c += DN)
            pv_tile(S, Nc, Vp + c, dp, Op + i * dp + c, dp, N, inv);
    }

    for (int i = 0; i < N; i++)
        memcpy(O + (size_t)i * d, Op + (size_t)i * dp, d * sizeof(float));
    free(Qp);
    free(Kp);
    free(Vp);
    free(Op);
    free(S);
}

/* ------------------------------------------------------------------------- */
/* Driver: do not change below this line.                                     */
/* ------------------------------------------------------------------------- */
#define REPS 5

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s <input.bin> <output.bin>\n", argv[0]);
        return 1;
    }
    FILE *in = fopen(argv[1], "rb");
    if (!in) {
        perror(argv[1]);
        return 1;
    }
    int32_t hdr[2];
    if (fread(hdr, sizeof hdr, 1, in) != 1) {
        fprintf(stderr, "%s: bad header\n", argv[1]);
        return 1;
    }
    int N = hdr[0], d = hdr[1];
    size_t n = (size_t)N * d;
    float *Q = malloc(n * sizeof(float)), *K = malloc(n * sizeof(float));
    float *V = malloc(n * sizeof(float)), *O = malloc(n * sizeof(float));
    if (!Q || !K || !V || !O || fread(Q, sizeof(float), n, in) != n ||
        fread(K, sizeof(float), n, in) != n || fread(V, sizeof(float), n, in) != n) {
        fprintf(stderr, "%s: could not read Q, K, V\n", argv[1]);
        return 1;
    }
    fclose(in);

    attention(Q, K, V, O, N, d); /* warm-up, not timed */
    double t[REPS];
    for (int r = 0; r < REPS; r++) {
        double t0 = now();
        attention(Q, K, V, O, N, d);
        t[r] = now() - t0;
    }
    qsort(t, REPS, sizeof t[0], cmp_double);
    double median = t[REPS / 2];

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    const char *name = strrchr(argv[0], '/') ? strrchr(argv[0], '/') + 1 : argv[0];
    printf("%-18s N=%-5d d=%-4d threads=%-3d time_ms=%10.3f  GFLOP/s=%8.2f\n", name, N, d,
           threads, median * 1e3, 4.0 * N * (double)N * d / median * 1e-9);

    FILE *out = fopen(argv[2], "wb");
    if (!out || fwrite(hdr, sizeof hdr, 1, out) != 1 || fwrite(O, sizeof(float), n, out) != n) {
        perror(argv[2]);
        return 1;
    }
    fclose(out);
    free(Q);
    free(K);
    free(V);
    free(O);
    return 0;
}
