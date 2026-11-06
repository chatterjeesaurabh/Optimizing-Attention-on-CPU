/*
 * attn_best.c -- Q3d: your fastest single-threaded version. Use any technique you like.
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
 * FlashAttention-style kernel written with AVX-512 intrinsics.
 *
 *  - Blocking: queries in blocks of RB = 96 rows, keys in blocks of BC rows
 *    where BC x d x 4 B = 32 KB (BC = 128 for d = 64, 64 for d = 128), so a
 *    V block fits in L1. For each (query block, key block) pair:
 *    S = Q K^T (RB x BC, in L1/L2), online softmax, O += P V. The K and V
 *    blocks are reused by all RB rows, so K and V are streamed from L2/L3
 *    only N / RB times; nothing of size N x N exists.
 *  - Online softmax: per row keep the running max m and sum l; when a new
 *    key block raises the max, the old O row and l are rescaled by
 *    2^(m_old - m_new). The rescale is folded into the PV kernel's
 *    accumulator load and the final 1/l is applied to the d outputs. The
 *    block's row max is taken inside the QK kernel while the scores are
 *    still in registers, and l is kept as a 16-lane vector, so the softmax
 *    is a single pass over S.
 *  - Vectorised exp: Q is prescaled by log2(e)/sqrt(d), so softmax needs
 *    2^x, computed 16 lanes at a time: x = n + f (n = round(x), |f| <= 1/2),
 *    2^f by a fitted degree-5 polynomial (rel. error 1.7e-7), times 2^n with
 *    vscalefps. glibc's scalar expf was the bottleneck of attn_fused.c.
 *  - Register tiles: QK^T uses 12 rows x 32 keys (24 zmm accumulators, per
 *    k: 2 K loads + 12 broadcasts for 24 FMAs); PV uses 6 rows x 64 columns
 *    (24 zmm accumulators, per key: 4 V loads + 6 broadcasts for 24 FMAs).
 *    24 independent chains cover the 4-cycle FMA latency on both FMA ports.
 *  - Packing: K is packed once into 32-key panels (d x 32) and Q (prescaled)
 *    into 12-row panels stored k-major (d x 12), so each kernel streams
 *    through contiguous memory with constant offsets; V is used in place.
 * Requires AVX-512F and d % 16 == 0; otherwise a portable scalar fallback runs.
 */
#include <immintrin.h>

#define QR 12  /* rows per QK^T register tile             */
#define KC 32  /* keys per QK^T register tile (2 vectors) */
#define PR 6   /* rows per PV register tile               */
#ifndef VBLOCK
#define VBLOCK 32768 /* bytes of V per key block (fits in L1D) */
#endif
#ifndef RB
#define RB 96  /* query rows per block (multiple of QR and PR) */
#endif

static size_t round_up(size_t x, size_t m) { return (x + m - 1) / m * m; }
static void *alloc64(size_t bytes) { return aligned_alloc(64, round_up(bytes, 64)); }

#if defined(__AVX512F__)
/* 2^x for x <= 0, 16 lanes */
static inline __m512 exp2_ps(__m512 x)
{
    x = _mm512_max_ps(x, _mm512_set1_ps(-126.0f));
    const __m512 n = _mm512_roundscale_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m512 f = _mm512_sub_ps(x, n);
    __m512 p = _mm512_set1_ps(1.34004322e-3f);
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(9.67603710e-3f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(5.55032721e-2f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(2.40221074e-1f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(6.93147207e-1f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.00000008e+0f));
    return _mm512_scalef_ps(p, n);
}

static inline __mmask16 tail_mask(int n) { return n >= 16 ? 0xFFFF : n <= 0 ? 0 : (__mmask16)((1u << n) - 1); }

/* S[0..QR)[0..KC) = Qs[0..QR)[0..d) * Kp (d x KC panel), S has row stride lds.
 * Also folds the first nvalid scores of each row into the running row max Mx[r]
 * while they are still in registers (saves a pass over S in the softmax). */
static inline void qk_kernel(const float *restrict Qs, const float *restrict Kp, float *restrict S, int lds,
                             int d, __m512 *restrict Mx, int nvalid)
{
    __m512 acc[QR][2];
    {
        const __m512 b0 = _mm512_load_ps(Kp), b1 = _mm512_load_ps(Kp + 16);
        for (int r = 0; r < QR; r++) {
            const __m512 a = _mm512_set1_ps(Qs[r]);
            acc[r][0] = _mm512_mul_ps(a, b0);
            acc[r][1] = _mm512_mul_ps(a, b1);
        }
    }
    for (int k = 1; k < d; k++) {
        const __m512 b0 = _mm512_load_ps(Kp + k * KC), b1 = _mm512_load_ps(Kp + k * KC + 16);
        for (int r = 0; r < QR; r++) {
            const __m512 a = _mm512_set1_ps(Qs[k * QR + r]);
            acc[r][0] = _mm512_fmadd_ps(a, b0, acc[r][0]);
            acc[r][1] = _mm512_fmadd_ps(a, b1, acc[r][1]);
        }
    }
    const __mmask16 k0 = tail_mask(nvalid), k1 = tail_mask(nvalid - 16);
    for (int r = 0; r < QR; r++) {
        _mm512_store_ps(S + r * lds, acc[r][0]);
        _mm512_store_ps(S + r * lds + 16, acc[r][1]);
        Mx[r] = _mm512_mask_max_ps(Mx[r], k0, Mx[r], acc[r][0]);
        Mx[r] = _mm512_mask_max_ps(Mx[r], k1, Mx[r], acc[r][1]);
    }
}

/* Ot[0..PR)[0..16W) = Ot * c[r] + P[0..PR)[0..nj) * V[0..nj)[0..16W); P has row stride lds */
static inline __attribute__((always_inline)) void pv_kernel(const float *restrict P, int lds, const float *restrict V,
                                                            int ldv, float *restrict Ot, int ldo, int nj,
                                                            const float *restrict c, const int W)
{
    __m512 acc[PR][4];
    for (int r = 0; r < PR; r++)
        for (int w = 0; w < W; w++)
            acc[r][w] = _mm512_mul_ps(_mm512_load_ps(Ot + r * ldo + 16 * w), _mm512_set1_ps(c[r]));
    for (int j = 0; j < nj; j++) {
        __m512 v[4];
        for (int w = 0; w < W; w++)
            v[w] = _mm512_loadu_ps(V + (size_t)j * ldv + 16 * w);
        for (int r = 0; r < PR; r++) {
            const __m512 p = _mm512_set1_ps(P[r * lds + j]);
            for (int w = 0; w < W; w++)
                acc[r][w] = _mm512_fmadd_ps(p, v[w], acc[r][w]);
        }
    }
    for (int r = 0; r < PR; r++)
        for (int w = 0; w < W; w++)
            _mm512_store_ps(Ot + r * ldo + 16 * w, acc[r][w]);
}

/* Online-softmax step for one row of a key block: s[0..nb) -> p = 2^(s - m_new) in place.
 * vmax holds the block's row max (from qk_kernel); the running sum L is kept as a
 * vector, so the only horizontal operation per row and block is one reduce_max. */
static inline void softmax_block(float *restrict s, int nb, __m512 vmax, float *m, __m512 *L, float *c)
{
    const float mnew = fmaxf(*m, _mm512_reduce_max_ps(vmax));
    const __m512 vm = _mm512_set1_ps(mnew);
    __m512 vsum = _mm512_setzero_ps();
    for (int j = 0; j < nb; j += 16) {
        const __m512 p = _mm512_maskz_mov_ps(tail_mask(nb - j), exp2_ps(_mm512_sub_ps(_mm512_load_ps(s + j), vm)));
        _mm512_store_ps(s + j, p);
        vsum = _mm512_add_ps(vsum, p);
    }
    *c = exp2f(*m - mnew); /* 0 on the first block (m = -inf) */
    *L = _mm512_fmadd_ps(*L, _mm512_set1_ps(*c), vsum);
    *m = mnew;
}

static void attention_avx512(const float *restrict Q, const float *restrict K, const float *restrict V,
                             float *restrict O, int N, int d)
{
    const float qscale = 1.4426950408889634f / sqrtf((float)d); /* log2(e) / sqrt(d) */
    const size_t Np = round_up(N, QR), Nc = round_up(N, KC);
    float *Qs = alloc64(Np * d * sizeof(float));
    float *Kp = alloc64(Nc * d * sizeof(float));
    const int BC = (int)round_up(VBLOCK / 4 / d > KC ? VBLOCK / 4 / d : KC, KC); /* keys per block */
    float *S = alloc64((size_t)RB * BC * sizeof(float));
    float *Ob = alloc64((size_t)RB * d * sizeof(float));
    float m[RB], c[RB];
    __m512 Mx[RB], L[RB];

    memset(Qs + (Np - QR) * d, 0, (size_t)QR * d * sizeof(float)); /* last (padded) panel */
    for (int i = 0; i < N; i++)
        for (int k = 0; k < d; k++)
            Qs[(size_t)(i / QR) * QR * d + (size_t)k * QR + i % QR] = Q[(size_t)i * d + k] * qscale;
    memset(Kp + (Nc - KC) * d, 0, (size_t)KC * d * sizeof(float));
    for (int j = 0; j < N; j++)
        for (int k = 0; k < d; k++)
            Kp[(size_t)(j / KC) * KC * d + (size_t)k * KC + j % KC] = K[(size_t)j * d + k];

    for (size_t i0 = 0; i0 < Np; i0 += RB) {
        const int rb = (int)(i0 + RB < Np ? RB : Np - i0); /* multiple of QR */
        for (int r = 0; r < rb; r++)
            m[r] = -INFINITY, L[r] = _mm512_setzero_ps();
        memset(Ob, 0, (size_t)rb * d * sizeof(float));

        for (int j0 = 0; j0 < N; j0 += BC) {
            const int nb = N - j0 < BC ? N - j0 : BC;
            for (int r = 0; r < rb; r++)
                Mx[r] = _mm512_set1_ps(-INFINITY);
            for (int r = 0; r < rb; r += QR)
                for (int j = 0; j < nb; j += KC)
                    qk_kernel(Qs + (i0 + r) * d, Kp + (size_t)(j0 + j) * d, S + r * BC + j, BC, d, Mx + r, nb - j);
            for (int r = 0; r < rb; r++)
                softmax_block(S + r * BC, nb, Mx[r], &m[r], &L[r], &c[r]);
            for (int r = 0; r < rb; r += PR) {
                int col = 0;
                for (; col + 64 <= d; col += 64)
                    pv_kernel(S + r * BC, BC, V + (size_t)j0 * d + col, d, Ob + r * d + col, d, nb, c + r, 4);
                for (; col < d; col += 16)
                    pv_kernel(S + r * BC, BC, V + (size_t)j0 * d + col, d, Ob + r * d + col, d, nb, c + r, 1);
            }
        }

        for (int r = 0; r < rb && i0 + r < (size_t)N; r++) {
            const __m512 inv = _mm512_set1_ps(1.0f / _mm512_reduce_add_ps(L[r]));
            for (int k = 0; k < d; k += 16)
                _mm512_storeu_ps(O + (i0 + r) * d + k, _mm512_mul_ps(_mm512_load_ps(Ob + r * d + k), inv));
        }
    }
    free(Qs);
    free(Kp);
    free(S);
    free(Ob);
}
#endif

/* Portable fallback: row-by-row fused attention. */
static void attention_scalar(const float *restrict Q, const float *restrict K, const float *restrict V,
                             float *restrict O, int N, int d)
{
    const float scale = 1.0f / sqrtf((float)d);
    float *s = malloc((size_t)N * sizeof(float));
    for (int i = 0; i < N; i++) {
        float max = -INFINITY, sum = 0.0f;
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < d; k++)
                acc += Q[(size_t)i * d + k] * K[(size_t)j * d + k];
            s[j] = acc * scale;
            max = s[j] > max ? s[j] : max;
        }
        for (int j = 0; j < N; j++)
            sum += s[j] = expf(s[j] - max);
        float *o = O + (size_t)i * d;
        for (int k = 0; k < d; k++)
            o[k] = 0.0f;
        for (int j = 0; j < N; j++)
            for (int k = 0; k < d; k++)
                o[k] += s[j] * V[(size_t)j * d + k];
        for (int k = 0; k < d; k++)
            o[k] /= sum;
    }
    free(s);
}

void attention(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
#if defined(__AVX512F__)
    if (d % 16 == 0) {
        attention_avx512(Q, K, V, O, N, d);
        return;
    }
#endif
    attention_scalar(Q, K, V, O, N, d);
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
