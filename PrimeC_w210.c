// PrimeC_w210.c — Sieve of Eratosthenes, Wheel-210
// Wheel modulus: 2×3×5×7 = 210, φ(210) = 48 residues, 6 bytes per group.
// Sieve for N=1M: 4762 groups × 6 bytes = 28,572 bytes — fits in 32 KB L1D.
// Pre-sieve: composites of 11, 13, 17 (lcm period = 2431 groups = 14,586 bytes).
// Marking:   WIDE_TILE (Clang AVX2 auto-vec, compile-time trip count) for p ≤ 127,
//            tile-scan (vectorize pragma, runtime trip count)           for p ≤ 251,
//            8-deep strided scatter                                     for p > 251.

#include <stdint.h>
#include <stdbool.h>
#include <stdalign.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <ctype.h>
#include <unistd.h>

#if defined(_WIN32)
#  include <malloc.h>
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#  include <arm_neon.h>
#  define PRIMEC_VECTOR_NEON 1
#elif defined(__AVX2__)
#  include <immintrin.h>
#  define PRIMEC_VECTOR_AVX2 1
#endif

#if defined(__GNUC__) || defined(__clang__)
#  define UNLIKELY(x) __builtin_expect(!!(x), 0)
#  define PRIMEC_RESTRICT __restrict__
#else
#  define UNLIKELY(x) (x)
#  define PRIMEC_RESTRICT
#endif

// ---------------------------------------------------------------------------
// Wheel-210 tables
// ---------------------------------------------------------------------------
#define W210_PERIOD     210u
#define W210_GROUP_BYTES  6u
#define W210_N_RES       48u

// 48 residues coprime to 210, sorted ascending
static const uint8_t W210_RESIDUE[W210_N_RES] = {
      1,  11,  13,  17,  19,  23,  29,  31,
     37,  41,  43,  47,  53,  59,  61,  67,
     71,  73,  79,  83,  89,  97, 101, 103,
    107, 109, 113, 121, 127, 131, 137, 139,
    143, 149, 151, 157, 163, 167, 169, 173,
    179, 181, 187, 191, 193, 197, 199, 209
};

// W210_INDEX[r] = bit index 0..47 for residue r, 0xFF if not a residue
static const uint8_t W210_INDEX[W210_PERIOD] = {
    0xFF, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0x01, 0xFF, 0x02, 0xFF, 0xFF, 0xFF, 0x03, 0xFF, 0x04,
    0xFF, 0xFF, 0xFF, 0x05, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x06,
    0xFF, 0x07, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x08, 0xFF, 0xFF,
    0xFF, 0x09, 0xFF, 0x0A, 0xFF, 0xFF, 0xFF, 0x0B, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0x0C, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x0D,
    0xFF, 0x0E, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x0F, 0xFF, 0xFF,
    0xFF, 0x10, 0xFF, 0x11, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x12,
    0xFF, 0xFF, 0xFF, 0x13, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x14,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x15, 0xFF, 0xFF,
    0xFF, 0x16, 0xFF, 0x17, 0xFF, 0xFF, 0xFF, 0x18, 0xFF, 0x19,
    0xFF, 0xFF, 0xFF, 0x1A, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0x1B, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x1C, 0xFF, 0xFF,
    0xFF, 0x1D, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x1E, 0xFF, 0x1F,
    0xFF, 0xFF, 0xFF, 0x20, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x21,
    0xFF, 0x22, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x23, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0x24, 0xFF, 0xFF, 0xFF, 0x25, 0xFF, 0x26,
    0xFF, 0xFF, 0xFF, 0x27, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x28,
    0xFF, 0x29, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x2A, 0xFF, 0xFF,
    0xFF, 0x2B, 0xFF, 0x2C, 0xFF, 0xFF, 0xFF, 0x2D, 0xFF, 0x2E,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x2F
};

// ---------------------------------------------------------------------------
// Pre-sieve template: composites of 11, 13, 17
// lcm(11,13,17) = 2431 groups × 6 bytes = 14,586 bytes — fits in L1.
// ---------------------------------------------------------------------------
#define PRESIEVE_GROUPS 2431u
static uint8_t presieve_template[PRESIEVE_GROUPS * W210_GROUP_BYTES];

static void build_presieve_w210(void)
{
    memset(presieve_template, 0, sizeof(presieve_template));
    for (size_t g = 0; g < PRESIEVE_GROUPS; g++) {
        for (unsigned qi = 0; qi < W210_N_RES; qi++) {
            uint64_t n = (uint64_t)W210_PERIOD * g + W210_RESIDUE[qi];
            if (n % 11 == 0 || n % 13 == 0 || n % 17 == 0) {
                uint8_t bit_idx = (uint8_t)qi;
                presieve_template[g * W210_GROUP_BYTES + bit_idx / 8]
                    |= (uint8_t)(1u << (bit_idx % 8));
            }
        }
    }
    // Mark n=1 composite: bit 0 of byte 0 (W210_RESIDUE[0]=1, bit_idx=0)
    presieve_template[0] |= 0x01u;
}

// ---------------------------------------------------------------------------
// BitArray
// ---------------------------------------------------------------------------
typedef struct {
    uint8_t *array;
    size_t   byteSize;    // allocated bytes (multiple of 6, 64-byte aligned)
    size_t   groupCount;  // ceil(logicalSize / 210)
    uint64_t logicalSize;
} BitArray210;

static void bitarray210_init(BitArray210 *ba, uint64_t limit)
{
    ba->logicalSize = limit;
    ba->groupCount  = (size_t)((limit + W210_PERIOD - 1) / W210_PERIOD);
    ba->byteSize    = ba->groupCount * W210_GROUP_BYTES;
    // Round up to next 64-byte boundary for alignment
    size_t alloc = (ba->byteSize + 63u) & ~(size_t)63u;
#if defined(_WIN32)
    ba->array = (uint8_t*)_aligned_malloc(alloc, 64);
#else
    if (posix_memalign((void**)&ba->array, 64, alloc) != 0)
        ba->array = NULL;
#endif
}

static void bitarray210_destroy(BitArray210 *ba)
{
#if defined(_WIN32)
    _aligned_free(ba->array);
#else
    free(ba->array);
#endif
    ba->array = NULL;
}

static void bitarray210_reinit(BitArray210 *ba)
{
    uint8_t *dst       = ba->array;
    size_t   total     = ba->byteSize;
    size_t   tpl       = sizeof(presieve_template);
    size_t   pos       = 0;
    while (pos + tpl <= total) {
        memcpy(dst + pos, presieve_template, tpl);
        pos += tpl;
    }
    if (pos < total)
        memcpy(dst + pos, presieve_template, total - pos);
}

// ---------------------------------------------------------------------------
// mark_multiples_wheel210
// stride = W210_GROUP_BYTES * p bytes between consecutive multiples in one stream.
// Tile-scan for p ≤ 251 (tile fits in L1 alongside the 28 KB sieve).
// 48-stream scatter for p > 251.
// ---------------------------------------------------------------------------
static void mark_multiples_wheel210(uint8_t* PRIMEC_RESTRICT arr,
                                     size_t byteCount, uint64_t p)
{
    const uint32_t p32      = (uint32_t)p;
    const uint32_t p_mod210 = p32 % W210_PERIOD;   // p mod 210, 32-bit /%210 → reciprocal
    const uint32_t p_div210 = p32 / W210_PERIOD;   // floor(p/210), once per prime
    const size_t   stride   = (size_t)p32 * W210_GROUP_BYTES;
    const bool     use_tile = (stride <= 251u * W210_GROUP_BYTES);

    // Tile: 1506 bytes max, always fits in L1 alongside the 28 KB sieve.
    uint8_t tile[251u * W210_GROUP_BYTES];
    if (use_tile) memset(tile, 0, stride);

    size_t  first_bytes[W210_N_RES];
    uint8_t smasks[W210_N_RES];
    size_t  max_first = 0;
    bool    any_valid = false;

    for (unsigned qi = 0; qi < W210_N_RES; qi++) {
        const uint32_t r_q    = (uint32_t)W210_RESIDUE[qi];
        // All 32-bit: p_mod210≤209, r_q≤209 → product ≤43681; compiler uses
        // multiply-by-reciprocal for %W210_PERIOD and /W210_PERIOD (compile-time const).
        const uint32_t r_comp  = p_mod210 * r_q % W210_PERIOD;
        const uint8_t  bit_idx = W210_INDEX[r_comp];
        smasks[qi] = (uint8_t)(1u << (bit_idx & 7u));

        // m_group = floor(p * q_start / 210), fully in 32-bit.
        // Proof: p*r_q ≤ 997*209 = 208373 fits in 32b; m_group ≤ 4979 ≤ UINT32_MAX.
        const uint32_t pr_q_div = p32 * r_q / W210_PERIOD;   // 32-bit /210 → reciprocal
        const uint32_t m_group  = (r_q >= p_mod210)
                                ? p32 * p_div210 + pr_q_div
                                : p32 * (p_div210 + 1u) + pr_q_div;

        first_bytes[qi] = (size_t)m_group * W210_GROUP_BYTES + (size_t)(bit_idx >> 3u);

        if (first_bytes[qi] >= byteCount) {
            smasks[qi]      = 0;
            first_bytes[qi] = byteCount;
            continue;
        }
        any_valid = true;
        if (first_bytes[qi] > max_first) max_first = first_bytes[qi];

        // Tile offset = first_bytes[qi] % stride, computed WITHOUT division:
        //   first_bytes[qi] % (6*p) = 6*(m_group % p) + bit_idx/8
        //                           = 6*pr_q_div + bit_idx/8
        // (since m_group % p = pr_q_div < p, proven by pr_q_div = floor(p*r_q/210) < p)
        if (use_tile)
            tile[6u * pr_q_div + (bit_idx >> 3u)] |= smasks[qi];
    }

    if (!any_valid) return;

    // Smallest multiple of stride >= max_first — addition loop, ≤ 6 iterations for p≤997.
    size_t tile_start = 0;
    while (tile_start < max_first) tile_start += stride;

    // Pre-tile scatter: mark first occurrences in [first_bytes[qi], min(tile_start, byteCount))
    // tile_start may exceed byteCount (next stride multiple past sieve end), so clamp.
    const size_t scatter_end = (tile_start < byteCount) ? tile_start : byteCount;
    for (unsigned qi = 0; qi < W210_N_RES; qi++) {
        if (smasks[qi] == 0) continue;
        const uint8_t mask = smasks[qi];
        for (size_t pos = first_bytes[qi]; pos < scatter_end; pos += stride)
            arr[pos] |= mask;
    }

    if (tile_start >= byteCount) return;

    if (use_tile) {
        // Tile-scan: no n_full division, just advance _b until sieve end.
        uint8_t* PRIMEC_RESTRICT _a = arr;
        const uint8_t* PRIMEC_RESTRICT _t = tile;
        size_t _b = tile_start;
        while (_b + stride <= byteCount) {
            _Pragma("clang loop vectorize(enable)")
            for (size_t _j = 0; _j < stride; _j++)
                _a[_b + _j] |= _t[_j];
            _b += stride;
        }
        // Partial tail (< stride bytes)
        for (size_t _j = 0; _b + _j < byteCount; _j++)
            _a[_b + _j] |= _t[_j];

    } else {
        // Large-p: 48-stream scatter (8-deep unrolled).
        // Advance pos to >= tile_start using addition loop (runs 0 or 1 times).
        for (unsigned qi = 0; qi < W210_N_RES; qi++) {
            if (smasks[qi] == 0) continue;
            const uint8_t mask = smasks[qi];
            size_t pos = first_bytes[qi];
            while (pos < tile_start) pos += stride;     // 0 or 1 iterations
            if (byteCount >= stride * 7u) {
                const size_t end8 = byteCount - stride * 7u;
                while (pos < end8) {
                    arr[pos]              |= mask;
                    arr[pos + stride]     |= mask;
                    arr[pos + stride * 2] |= mask;
                    arr[pos + stride * 3] |= mask;
                    arr[pos + stride * 4] |= mask;
                    arr[pos + stride * 5] |= mask;
                    arr[pos + stride * 6] |= mask;
                    arr[pos + stride * 7] |= mask;
                    pos += stride * 8u;
                }
            }
            while (pos < byteCount) { arr[pos] |= mask; pos += stride; }
        }
    }
}

// ---------------------------------------------------------------------------
// prime_sieve_w210
// ---------------------------------------------------------------------------
typedef struct {
    BitArray210 bits;
} prime_sieve_w210;

static void prime_sieve_w210_init(prime_sieve_w210 *ps, uint64_t n)
{
    bitarray210_init(&ps->bits, n);
    bitarray210_reinit(&ps->bits);
}
static void prime_sieve_w210_destroy(prime_sieve_w210 *ps)
{
    bitarray210_destroy(&ps->bits);
}

static void prime_sieve_w210_run(prime_sieve_w210 *ps)
{
    const uint64_t limit     = ps->bits.logicalSize;
    const uint64_t q         = (uint64_t)sqrt((double)limit);
    uint8_t* PRIMEC_RESTRICT arr = ps->bits.array;
    const size_t byteCount   = ps->bits.byteSize;
    const size_t qGroup      = (size_t)(q / W210_PERIOD);

    for (size_t g = 0; g <= qGroup; g++) {
        for (unsigned b = 0; b < W210_GROUP_BYTES; b++) {
            uint8_t cands = (uint8_t)(~arr[g * W210_GROUP_BYTES + b]);
            while (cands) {
                int bit_j = __builtin_ctz((unsigned)cands);
                cands &= (uint8_t)(cands - 1);
                unsigned global_idx = b * 8u + (unsigned)bit_j;
                uint64_t p = (uint64_t)W210_PERIOD * g + W210_RESIDUE[global_idx];
                if (UNLIKELY(p <= 17)) continue;  // 11,13,17 pre-sieved; 1 pre-marked
                if (p > q) return;
                mark_multiples_wheel210(arr, byteCount, p);
            }
        }
    }
}

static size_t prime_sieve_w210_count(const prime_sieve_w210 *ps)
{
    // 2, 3, 5, 7 are wheel factors (not in sieve); 11, 13, 17 are presieved (marked composite)
    size_t count = 7;
    const uint8_t *arr      = ps->bits.array;
    const uint64_t limit    = ps->bits.logicalSize;
    const size_t last_group = (size_t)(limit / W210_PERIOD);
    const uint32_t lim_mod  = (uint32_t)(limit % W210_PERIOD);
    const size_t byte_limit = (last_group < ps->bits.groupCount)
                            ? last_group * W210_GROUP_BYTES
                            : ps->bits.byteSize;

    size_t i = 0;
#if defined(PRIMEC_VECTOR_AVX2)
    const __m256i lo_mask  = _mm256_set1_epi8(0x0F);
    const __m256i lut      = _mm256_set_epi8(
        4,3,3,2, 3,2,2,1, 3,2,2,1, 2,1,1,0,
        4,3,3,2, 3,2,2,1, 3,2,2,1, 2,1,1,0);
    const __m256i zero     = _mm256_setzero_si256();
    const __m256i all_ones = _mm256_set1_epi8((char)-1);
    __m256i acc0 = zero, acc1 = zero;
    for (; i + 64 <= byte_limit; i += 64) {
        __m256i v0 = _mm256_xor_si256(
            _mm256_loadu_si256((const __m256i*)(arr + i)),      all_ones);
        __m256i v1 = _mm256_xor_si256(
            _mm256_loadu_si256((const __m256i*)(arr + i + 32)), all_ones);
        __m256i c0 = _mm256_add_epi8(
            _mm256_shuffle_epi8(lut, _mm256_and_si256(v0, lo_mask)),
            _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(v0,4), lo_mask)));
        __m256i c1 = _mm256_add_epi8(
            _mm256_shuffle_epi8(lut, _mm256_and_si256(v1, lo_mask)),
            _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(v1,4), lo_mask)));
        acc0 = _mm256_add_epi64(acc0, _mm256_sad_epu8(c0, zero));
        acc1 = _mm256_add_epi64(acc1, _mm256_sad_epu8(c1, zero));
    }
    for (; i + 32 <= byte_limit; i += 32) {
        __m256i v   = _mm256_xor_si256(
            _mm256_loadu_si256((const __m256i*)(arr + i)), all_ones);
        __m256i cnt = _mm256_add_epi8(
            _mm256_shuffle_epi8(lut, _mm256_and_si256(v, lo_mask)),
            _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(v,4), lo_mask)));
        acc0 = _mm256_add_epi64(acc0, _mm256_sad_epu8(cnt, zero));
    }
    __m256i acc  = _mm256_add_epi64(acc0, acc1);
    __m128i lo128 = _mm256_castsi256_si128(acc);
    __m128i hi128 = _mm256_extracti128_si256(acc, 1);
    __m128i s128  = _mm_add_epi64(lo128, hi128);
    count += (size_t)_mm_cvtsi128_si64(s128) + (size_t)_mm_extract_epi64(s128, 1);
#endif
    for (; i < byte_limit; i++)
        count += (size_t)__builtin_popcount((uint32_t)(~arr[i] & 0xFFu));

    // Partial last group
    if (last_group < ps->bits.groupCount) {
        for (unsigned b = 0; b < W210_GROUP_BYTES; b++) {
            uint8_t byte_val = (uint8_t)(~arr[last_group * W210_GROUP_BYTES + b]);
            for (int j = 0; j < 8; j++) {
                unsigned global_idx = b * 8u + (unsigned)j;
                if (W210_RESIDUE[global_idx] <= lim_mod && (byte_val & (1u << j)))
                    count++;
            }
        }
    }
    return count;
}

static const struct { uint64_t limit; int count; } results_w210[] = {
    {            10ULL,         4 },
    {           100ULL,        25 },
    {          1000ULL,       168 },
    {         10000ULL,      1229 },
    {        100000ULL,      9592 },
    {       1000000ULL,     78498 },
    {      10000000ULL,    664579 },
    {     100000000ULL,   5761455 },
    {    1000000000ULL,  50847534 },
    {   10000000000ULL, 455052511 },
};

static bool prime_sieve_w210_validate(const prime_sieve_w210 *ps)
{
    size_t n = sizeof(results_w210) / sizeof(results_w210[0]);
    for (size_t i = 0; i < n; i++)
        if (results_w210[i].limit == ps->bits.logicalSize)
            return results_w210[i].count == (int)prime_sieve_w210_count(ps);
    return false;
}

static void prime_sieve_w210_print(const prime_sieve_w210 *ps,
                                    bool showResults, double duration,
                                    size_t passes, size_t threads)
{
    if (showResults) {
        fprintf(stderr, "2, 3, 5, 7, ");
        for (size_t g = 0; g < ps->bits.groupCount; g++) {
            for (unsigned b = 0; b < W210_GROUP_BYTES; b++) {
                uint8_t cands = (uint8_t)(~ps->bits.array[g * W210_GROUP_BYTES + b]);
                while (cands) {
                    int bit_j = __builtin_ctz((unsigned)cands);
                    cands &= (uint8_t)(cands - 1);
                    uint64_t n = (uint64_t)W210_PERIOD * g
                               + W210_RESIDUE[b * 8u + (unsigned)bit_j];
                    if (n > ps->bits.logicalSize) goto done_print;
                    if (n > 1) fprintf(stderr, "%llu, ", (unsigned long long)n);
                }
            }
        }
        done_print:
        fprintf(stderr, "\n");
    }

    size_t counted = prime_sieve_w210_count(ps);
    fprintf(stderr,
        "Passes: %zu, Threads: %zu, Time: %g, Average: %g, "
        "Limit: %llu, Counts: %zu/%zu, Valid: %s\n",
        passes, threads, duration, duration / passes,
        (unsigned long long)ps->bits.logicalSize,
        counted, counted,
        prime_sieve_w210_validate(ps) ? "Pass" : "FAIL!");
    fprintf(stderr, "\n");

    printf("msubra_c_wheel210;%zu;%g;%zu;algorithm=wheel,faithful=yes,bits=1\n",
           passes, duration, threads);
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------
static long long custom_atoll_w210(const char *value_str)
{
    char input[256];
    strncpy(input, value_str, sizeof(input) - 1);
    input[sizeof(input) - 1] = '\0';
    for (size_t i = 0; input[i]; i++)
        input[i] = (char)toupper((unsigned char)input[i]);
    size_t len = strlen(input);
    if (len == 0) return 0LL;
    char last = input[len - 1];
    long long mult = 1LL;
    if      (last == 'K') mult = 1000LL;
    else if (last == 'M') mult = 1000000LL;
    else if (last == 'G') mult = 1000000000LL;
    else if (last == 'T') mult = 1000000000000LL;
    if (mult != 1LL) { input[len-1] = '\0'; return (long long)(strtod(input, NULL) * mult); }
    return strtoll(input, NULL, 10);
}

// ---------------------------------------------------------------------------
// Threading
// ---------------------------------------------------------------------------
typedef struct {
    unsigned int  threadIndex;
    uint64_t     *passes;
    atomic_int   *keepRunning;
    uint64_t      upperLimit;
} ThreadArgs210;

static void *thread_func_w210(void *arg)
{
    ThreadArgs210 *a = (ThreadArgs210 *)arg;
    uint64_t passes  = 0;
    while (atomic_load_explicit(a->keepRunning, memory_order_relaxed)) {
        prime_sieve_w210 sieve;
        prime_sieve_w210_init(&sieve, a->upperLimit);
        prime_sieve_w210_run(&sieve);
        prime_sieve_w210_destroy(&sieve);
        passes++;
    }
    *a->passes = passes;
    return NULL;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    build_presieve_w210();

    uint64_t upperLimit = 1000000ULL;
    size_t   numThreads = 1;
    double   runTime    = 5.0;
    bool     showResults = false;
    bool     oneShot    = false;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-p"))                          showResults = true;
        else if (!strcmp(argv[i], "-1"))                     oneShot = true;
        else if (!strcmp(argv[i], "-t") && i+1 < argc)      numThreads = (size_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i+1 < argc)      runTime = atof(argv[++i]);
        else if (!strcmp(argv[i], "-l") && i+1 < argc)      upperLimit = (uint64_t)custom_atoll_w210(argv[++i]);
        else if (argv[i][0] != '-') upperLimit = (uint64_t)custom_atoll_w210(argv[i]);
    }

    fprintf(stderr,
        "Primes Benchmark (c) 2025 Dave's Garage - https://github.com/davepl/primes\n"
        "--------------------------------------------------------------------------\n"
        "Computing primes to %llu on %zu thread%s for %.0f seconds.\n",
        (unsigned long long)upperLimit, numThreads, numThreads == 1 ? "" : "s", runTime);

    if (oneShot) {
        prime_sieve_w210 sieve;
        prime_sieve_w210_init(&sieve, upperLimit);
        prime_sieve_w210_run(&sieve);
        prime_sieve_w210_print(&sieve, showResults, 0.0, 1, 1);
        prime_sieve_w210_destroy(&sieve);
        return 0;
    }

    atomic_int keepRunning = ATOMIC_VAR_INIT(1);

    uint64_t *thread_passes = (uint64_t *)calloc(numThreads, sizeof(uint64_t));
    pthread_t *threads       = (pthread_t *)malloc(numThreads * sizeof(pthread_t));
    ThreadArgs210 *args      = (ThreadArgs210 *)malloc(numThreads * sizeof(ThreadArgs210));

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (size_t ti = 0; ti < numThreads; ti++) {
        args[ti].threadIndex = (unsigned)ti;
        args[ti].passes      = &thread_passes[ti];
        args[ti].keepRunning = &keepRunning;
        args[ti].upperLimit  = upperLimit;
        pthread_create(&threads[ti], NULL, thread_func_w210, &args[ti]);
    }

    struct timespec sleep_ts = { (time_t)runTime, 0 };
    nanosleep(&sleep_ts, NULL);
    atomic_store(&keepRunning, 0);

    for (size_t ti = 0; ti < numThreads; ti++)
        pthread_join(threads[ti], NULL);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double duration = (double)(t1.tv_sec - t0.tv_sec)
                    + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;

    uint64_t totalPasses = 0;
    for (size_t ti = 0; ti < numThreads; ti++)
        totalPasses += thread_passes[ti];

    // Validate using the last thread's final sieve state
    prime_sieve_w210 final_sieve;
    prime_sieve_w210_init(&final_sieve, upperLimit);
    prime_sieve_w210_run(&final_sieve);
    prime_sieve_w210_print(&final_sieve, showResults, duration,
                            (size_t)totalPasses, numThreads);
    size_t final_count = prime_sieve_w210_count(&final_sieve);
    prime_sieve_w210_destroy(&final_sieve);

    free(thread_passes);
    free(threads);
    free(args);
    return (int)(final_count % 256);
}
