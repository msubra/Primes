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

// ---------------------------------------------------------------------------
// SIMD detection
// ---------------------------------------------------------------------------
#if defined(__aarch64__) || defined(_M_ARM64)
#  include <arm_neon.h>
#  define PRIMEC_VECTOR_NEON 1
#elif defined(__AVX512F__)
#  include <immintrin.h>
#  define PRIMEC_VECTOR_AVX2 1
#elif defined(__AVX2__)
#  include <immintrin.h>
#  define PRIMEC_VECTOR_AVX2 1
#endif

// ---------------------------------------------------------------------------
// Tuning / compiler hint macros
// ---------------------------------------------------------------------------
static const uint64_t DEFAULT_UPPER_LIMIT = 1000000ULL;

#ifndef USE_BRANCH_HINTS
#  define USE_BRANCH_HINTS 1
#endif
#if USE_BRANCH_HINTS && (defined(__GNUC__) || defined(__clang__))
#  define LIKELY(x)   (__builtin_expect(!!(x), 1))
#  define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#else
#  define LIKELY(x)   (x)
#  define UNLIKELY(x) (x)
#endif

#if defined(__GNUC__) || defined(__clang__)
#  define ATTR_ALWAYS_INLINE __attribute__((always_inline))
#  define PRIMEC_RESTRICT    __restrict__
#else
#  define ATTR_ALWAYS_INLINE
#  define PRIMEC_RESTRICT
#endif

// ---------------------------------------------------------------------------
// Wheel of 30: store only numbers coprime to 30.
// 8 residues per wheel period: {1, 7, 11, 13, 17, 19, 23, 29}.
// Bit j of byte i represents n = 30*i + WHEEL30_RESIDUE[j].
// Convention: 0 = prime candidate, 1 = composite.
//
// Array size for limit N: N/30 + 2 bytes (~33 KB for N=1,000,000 vs 62.5 KB
// for odd-only storage).  Composites of 2, 3, 5 are never stored, so no
// presieve tile is needed.
// ---------------------------------------------------------------------------
static const uint8_t WHEEL30_RESIDUE[8] = {1, 7, 11, 13, 17, 19, 23, 29};
// WHEEL30_INDEX[r] = bit position for residue r; 0xFF if r not coprime to 30.
static const uint8_t WHEEL30_INDEX[30] = {
    0xFF, 0,    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 1,
    0xFF, 0xFF, 0xFF, 2,    0xFF, 3,    0xFF, 0xFF,
    0xFF, 4,    0xFF, 5,    0xFF, 0xFF, 0xFF, 6,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 7
};

// ---------------------------------------------------------------------------
// BitArray
// ---------------------------------------------------------------------------
typedef enum {
    ALLOC_NEW_ARRAY,
    ALLOC_POSIX_ALIGNED,
    ALLOC_WIN_ALIGNED
} AllocationKind;

typedef struct {
    uint8_t*       array;
    size_t         logicalSize;
    size_t         byteSize;
    AllocationKind allocationKind;
} BitArray;

// Wheel-30: 1 byte per 30 numbers; +2 for alignment safety.
static inline size_t ba_storage_size(size_t n) {
    return n / 30 + 2;
}

static void bitarray_init(BitArray* ba, size_t size)
{
    ba->logicalSize = size;
    ba->byteSize    = ba_storage_size(size);

#if defined(_WIN32)
    void* ptr = _aligned_malloc(ba->byteSize, 64);
    if (ptr) {
        ba->array           = (uint8_t*)ptr;
        ba->allocationKind  = ALLOC_WIN_ALIGNED;
    } else {
        ba->array           = (uint8_t*)malloc(ba->byteSize);
        ba->allocationKind  = ALLOC_NEW_ARRAY;
    }
#else
    void* ptr = NULL;
    if (posix_memalign(&ptr, 64, ba->byteSize) == 0 && ptr) {
        ba->array           = (uint8_t*)ptr;
        ba->allocationKind  = ALLOC_POSIX_ALIGNED;
    } else {
        ba->array           = (uint8_t*)malloc(ba->byteSize);
        ba->allocationKind  = ALLOC_NEW_ARRAY;
    }
#endif
    memset(ba->array, 0x00, ba->byteSize);
}

// Reinitialize without re-allocating — used in the hot benchmark loop.
// Zero the array then mark n=1 (byte 0, bit 0) as composite.
static inline void bitarray_reinit(BitArray* ba)
{
    memset(ba->array, 0, ba->byteSize);
    ba->array[0] = 0x01;   // n = 1 is not prime
}

static void bitarray_destroy(BitArray* ba)
{
    switch (ba->allocationKind) {
        case ALLOC_POSIX_ALIGNED: free(ba->array);           break;
#if defined(_WIN32)
        case ALLOC_WIN_ALIGNED:   _aligned_free(ba->array);  break;
#endif
        default:                  free(ba->array);           break;
    }
    ba->array = NULL;
}

// Returns true if n is prime.
// Handles 2, 3, 5 specially (not stored in array); multiples of 2/3/5 return false.
static inline bool ATTR_ALWAYS_INLINE bitarray_get(const BitArray* ba, size_t n)
{
    if (n == 2 || n == 3 || n == 5) return true;
    if (n < 7) return false;
    uint8_t bidx = WHEEL30_INDEX[n % 30];
    if (bidx == 0xFF) return false;          // divisible by 2, 3, or 5
    size_t byte_i = n / 30;
    if (byte_i >= ba->byteSize) return false;
    return !(ba->array[byte_i] & (uint8_t)(1u << bidx));
}

// ---------------------------------------------------------------------------
// Outlined AVX2 per-prime tile functions.
//
// Each function is __attribute__((noinline)) so it lives in its own region of
// the µop cache (DSB = 1536 µops on Coffee Lake).  Previously these loops were
// generated inline by a switch macro, making the entire mark_multiples_wheel30
// function too large for the DSB and evicting the hot small-prime paths.
// Outlining keeps the calling function compact while still giving each case a
// fully-unrolled inner loop with a compile-time stride constant.
// ---------------------------------------------------------------------------
#if defined(PRIMEC_VECTOR_AVX2)
#define DEFINE_AVX2_TILE(P) \
static __attribute__((noinline)) size_t \
avx2_tile_p##P(uint8_t* arr, size_t pos, size_t byteCount, const __m256i* nvecs) \
{ \
    while (pos + (size_t)(P) * 32u <= byteCount) { \
        _Pragma("clang loop unroll(full)") \
        for (int _i = 0; _i < (P); _i++) \
            _mm256_storeu_si256((__m256i*)(arr + pos + (size_t)_i * 32u), \
                _mm256_or_si256( \
                    _mm256_loadu_si256((const __m256i*)(arr + pos + (size_t)_i * 32u)), \
                    nvecs[_i])); \
        pos += (size_t)(P) * 32u; \
    } \
    return pos; \
}
DEFINE_AVX2_TILE( 7) DEFINE_AVX2_TILE(11) DEFINE_AVX2_TILE(13)
DEFINE_AVX2_TILE(17) DEFINE_AVX2_TILE(19) DEFINE_AVX2_TILE(23)
DEFINE_AVX2_TILE(29) DEFINE_AVX2_TILE(31) DEFINE_AVX2_TILE(37)
DEFINE_AVX2_TILE(41) DEFINE_AVX2_TILE(43) DEFINE_AVX2_TILE(47)
DEFINE_AVX2_TILE(53) DEFINE_AVX2_TILE(59) DEFINE_AVX2_TILE(61)
DEFINE_AVX2_TILE(67) DEFINE_AVX2_TILE(71) DEFINE_AVX2_TILE(73)
DEFINE_AVX2_TILE(79)
#undef DEFINE_AVX2_TILE
#endif  // PRIMEC_VECTOR_AVX2

// ---------------------------------------------------------------------------
// mark_multiples_wheel30
//
// For prime p (p >= 7, coprime to 30), marks all composites p*q where q >= p
// and q is coprime to 30.
//
// Small-p path (p <= 79):
//   Build a combined p-byte "tile" where tile[b % p] = OR of all series masks
//   active at byte offset b.  After a short startup region, apply the tile
//   with SIMD (AVX2: 32-byte stores; NEON: 16-byte stores), giving ~32x or
//   ~16x fewer store operations than byte-at-a-time.
//
// Large-p path (p > 79):
//   Fall back to 8 independent strided byte-write loops.
// ---------------------------------------------------------------------------
static void mark_multiples_wheel30(uint8_t* PRIMEC_RESTRICT arr,
                                    size_t byteCount, uint64_t p)
{
    const uint32_t p_mod30 = (uint32_t)(p % 30);
    const size_t   stride  = (size_t)p;

    // Compute per-series starting byte and bit mask
    size_t  first_bytes[8];
    uint8_t smasks[8];
    size_t  max_first = 0;
    bool    any_valid = false;

    for (int qi = 0; qi < 8; qi++) {
        const uint32_t r_q    = WHEEL30_RESIDUE[qi];
        const uint32_t r_comp = (p_mod30 * r_q) % 30;
        smasks[qi]            = (uint8_t)(1u << WHEEL30_INDEX[r_comp]);

        uint64_t q_start;
        if (r_q >= p_mod30)
            q_start = p - p_mod30 + r_q;
        else
            q_start = p - p_mod30 + 30 + r_q;

        first_bytes[qi] = (size_t)((p * q_start) / 30);
        if (first_bytes[qi] >= byteCount) {
            smasks[qi]     = 0;
            first_bytes[qi] = byteCount;
            continue;
        }
        any_valid = true;
        if (first_bytes[qi] > max_first) max_first = first_bytes[qi];
    }

    if (!any_valid) return;

#if defined(PRIMEC_VECTOR_NEON)
    // ---------------------------------------------------------------------------
    // NEON period-tile path for small primes (stride <= 61).
    //
    // All 8 series share byte-stride = p.  The combined OR mask repeats with
    // period p bytes.  Build a p-byte tile, expand it to 16p bytes (one NEON
    // vector per tile slot), then apply stride NEON OR ops per 16p-byte block.
    //
    // Key: use switch(stride) so each case gets a compile-time constant N,
    // allowing Clang -O3 to fully unroll `for (_i < N)` and keep all N nvec
    // registers live — eliminating inner-loop branch overhead entirely.
    // ---------------------------------------------------------------------------
    if (stride <= 61) {
        // Build the p-byte period tile.
        uint8_t tile[64];
        memset(tile, 0, stride);
        for (int qi = 0; qi < 8; qi++) {
            if (smasks[qi] == 0) continue;
            tile[first_bytes[qi] % stride] |= smasks[qi];
        }

        // tile_start: smallest multiple of stride that is >= max_first.
        const size_t tile_start = ((max_first + stride - 1) / stride) * stride;

        // Pre-tile: scalar marking for bytes [first_bytes[qi], tile_start).
        for (int qi = 0; qi < 8; qi++) {
            if (smasks[qi] == 0) continue;
            const uint8_t mask = smasks[qi];
            for (size_t pos = first_bytes[qi]; pos < tile_start; pos += stride)
                arr[pos] |= mask;
        }

        // Build bigtile (stride × 16 bytes) and load into nvecs array.
        uint8_t bigtile[61 * 16];
        for (size_t i = 0; i < stride * 16u; i++) bigtile[i] = tile[i % stride];
        uint8x16_t nvecs[61];
        for (size_t i = 0; i < stride; i++) nvecs[i] = vld1q_u8(&bigtile[i * 16u]);

        // Apply stride NEON ops per 16*stride-byte block.
        // The macro fixes the loop bound as a compile-time literal so Clang
        // fully unrolls the inner loop, keeping all nvecs in registers.
#define NEON_TILE_LOOP(P) \
        { \
            size_t _pos = tile_start; \
            while (_pos + (size_t)(P) * 16u <= byteCount) { \
                _Pragma("clang loop unroll(full)") \
                for (int _i = 0; _i < (P); _i++) \
                    vst1q_u8(&arr[_pos + (size_t)_i * 16u], \
                             vorrq_u8(vld1q_u8(&arr[_pos + (size_t)_i * 16u]), nvecs[_i])); \
                _pos += (size_t)(P) * 16u; \
            } \
            pos = _pos; \
        }

        size_t pos;
        switch (stride) {
            case  7: NEON_TILE_LOOP( 7); break;
            case 11: NEON_TILE_LOOP(11); break;
            case 13: NEON_TILE_LOOP(13); break;
            case 17: NEON_TILE_LOOP(17); break;
            case 19: NEON_TILE_LOOP(19); break;
            case 23: NEON_TILE_LOOP(23); break;
            case 29: NEON_TILE_LOOP(29); break;
            case 31: NEON_TILE_LOOP(31); break;
            case 37: NEON_TILE_LOOP(37); break;
            case 41: NEON_TILE_LOOP(41); break;
            case 43: NEON_TILE_LOOP(43); break;
            case 47: NEON_TILE_LOOP(47); break;
            case 53: NEON_TILE_LOOP(53); break;
            case 59: NEON_TILE_LOOP(59); break;
            case 61: NEON_TILE_LOOP(61); break;
            default: pos = tile_start; break;  // shouldn't happen
        }
#undef NEON_TILE_LOOP

        // Scalar cleanup for the remaining < stride*16 bytes.
        size_t tile_idx = 0;
        while (pos < byteCount) {
            arr[pos++] |= tile[tile_idx];
            if (++tile_idx == stride) tile_idx = 0;
        }
        return;
    }
#endif  // PRIMEC_VECTOR_NEON

#if defined(PRIMEC_VECTOR_AVX2)
    // ---------------------------------------------------------------------------
    // AVX2 period-tile path for small primes (stride <= 61).
    //
    // Same concept as the NEON path but with 256-bit YMM vectors (32 bytes).
    // On x86, the store unit handles one 256-bit store per cycle (vs 1 byte for
    // scalar), giving ~32× better store throughput.  For p=7 this replaces 37K
    // scalar stores with 1041 AVX256 stores — a 35× reduction in store ops.
    // ---------------------------------------------------------------------------
    // Extended-tile optimisation: instead of a stride×32-byte bigtile, build a
    // (stride+31)-byte extended_tile.  A 32-byte window starting at offset i
    // within an infinite periodic tiling is exactly extended_tile[i..i+31], so
    // nvecs[i] = loadu256(extended_tile + i).  Fill cost drops from stride×32
    // iterations to stride+31 iterations (21× cheaper at stride=61, 5× at stride=7).
    // ---------------------------------------------------------------------------
    if (stride <= 79) {
        uint8_t tile[80];
        memset(tile, 0, stride);
        for (int qi = 0; qi < 8; qi++) {
            if (smasks[qi] == 0) continue;
            tile[first_bytes[qi] % stride] |= smasks[qi];
        }

        const size_t tile_start = ((max_first + stride - 1) / stride) * stride;

        for (int qi = 0; qi < 8; qi++) {
            if (smasks[qi] == 0) continue;
            const uint8_t mask = smasks[qi];
            for (size_t pos = first_bytes[qi]; pos < tile_start; pos += stride)
                arr[pos] |= mask;
        }

        // Build extended_tile[stride+31]: stride+31 iterations instead of stride×32.
        // nvecs[n] = loadu256(extended_tile + (n*32)%stride) via rolling step.
        uint8_t extended_tile[79 + 31];
        { size_t t = 0;
          for (size_t i = 0; i < stride + 31u; i++) {
              extended_tile[i] = tile[t];
              if (++t == stride) t = 0;
          }
        }
        __m256i nvecs[79];
        { const size_t step = 32u % stride;
          size_t off = 0;
          for (size_t i = 0; i < stride; i++) {
              nvecs[i] = _mm256_loadu_si256((const __m256i*)(extended_tile + off));
              off += step;
              if (off >= stride) off -= stride;
          }
        }

        size_t pos;
        switch (stride) {
            case  7: pos = avx2_tile_p7 (arr, tile_start, byteCount, nvecs); break;
            case 11: pos = avx2_tile_p11(arr, tile_start, byteCount, nvecs); break;
            case 13: pos = avx2_tile_p13(arr, tile_start, byteCount, nvecs); break;
            case 17: pos = avx2_tile_p17(arr, tile_start, byteCount, nvecs); break;
            case 19: pos = avx2_tile_p19(arr, tile_start, byteCount, nvecs); break;
            case 23: pos = avx2_tile_p23(arr, tile_start, byteCount, nvecs); break;
            case 29: pos = avx2_tile_p29(arr, tile_start, byteCount, nvecs); break;
            case 31: pos = avx2_tile_p31(arr, tile_start, byteCount, nvecs); break;
            case 37: pos = avx2_tile_p37(arr, tile_start, byteCount, nvecs); break;
            case 41: pos = avx2_tile_p41(arr, tile_start, byteCount, nvecs); break;
            case 43: pos = avx2_tile_p43(arr, tile_start, byteCount, nvecs); break;
            case 47: pos = avx2_tile_p47(arr, tile_start, byteCount, nvecs); break;
            case 53: pos = avx2_tile_p53(arr, tile_start, byteCount, nvecs); break;
            case 59: pos = avx2_tile_p59(arr, tile_start, byteCount, nvecs); break;
            case 61: pos = avx2_tile_p61(arr, tile_start, byteCount, nvecs); break;
            case 67: pos = avx2_tile_p67(arr, tile_start, byteCount, nvecs); break;
            case 71: pos = avx2_tile_p71(arr, tile_start, byteCount, nvecs); break;
            case 73: pos = avx2_tile_p73(arr, tile_start, byteCount, nvecs); break;
            case 79: pos = avx2_tile_p79(arr, tile_start, byteCount, nvecs); break;
            default: pos = tile_start; break;
        }

        // SIMD cleanup: reuse nvecs for full 32-byte blocks, then scalar for the
        // sub-32-byte tail.  nvec_idx starts at 0 (pos is aligned to stride×32).
        // tile_off tracks the tile byte position for the final scalar bytes.
        { const size_t pf_step = 32u % stride;
          size_t ni = 0, tf = 0;
          while (pos + 32u <= byteCount) {
              _mm256_storeu_si256((__m256i*)(arr + pos),
                  _mm256_or_si256(_mm256_loadu_si256((const __m256i*)(arr + pos)),
                                  nvecs[ni]));
              pos += 32;
              if (++ni == stride) ni = 0;
              tf += pf_step;
              if (tf >= stride) tf -= stride;
          }
          while (pos < byteCount) {
              arr[pos++] |= tile[tf];
              if (++tf == stride) tf = 0;
          }
        }
        return;
    }
#endif  // PRIMEC_VECTOR_AVX2

    // 8 independent strided loops with 8-deep unrolling so the CPU can
    // pipeline all 8 independent load-OR-store ops in each outer iteration.
    for (int qi = 0; qi < 8; qi++) {
        if (smasks[qi] == 0) continue;
        const uint8_t mask = smasks[qi];
        size_t pos = first_bytes[qi];

        // Unrolled 8× body: 8 independent load-OR-store ops per outer iteration.
        if (byteCount >= stride * 7) {
            const size_t end8 = byteCount - stride * 7;
            while (pos < end8) {
                arr[pos]            |= mask;
                arr[pos + stride]   |= mask;
                arr[pos + stride*2] |= mask;
                arr[pos + stride*3] |= mask;
                arr[pos + stride*4] |= mask;
                arr[pos + stride*5] |= mask;
                arr[pos + stride*6] |= mask;
                arr[pos + stride*7] |= mask;
                pos += stride * 8;
            }
        }
        while (pos < byteCount) { arr[pos] |= mask; pos += stride; }
    }
}

// ---------------------------------------------------------------------------
// prime_sieve
// ---------------------------------------------------------------------------
typedef struct {
    BitArray bits;
} prime_sieve;

static void prime_sieve_init(prime_sieve* ps, uint64_t n) {
    bitarray_init(&ps->bits, (size_t)n);
    bitarray_reinit(&ps->bits);
}
static void prime_sieve_destroy(prime_sieve* ps) {
    bitarray_destroy(&ps->bits);
}

static void prime_sieve_run(prime_sieve* ps)
{
    const uint64_t limit   = ps->bits.logicalSize;
    const uint64_t q       = (uint64_t)sqrt((double)limit);
    uint8_t* PRIMEC_RESTRICT arr = ps->bits.array;
    const size_t byteCount = ps->bits.byteSize;
    const size_t qByte     = (size_t)(q / 30);

    for (size_t byte_i = 0; byte_i <= qByte; byte_i++) {
        uint8_t primes = (uint8_t)(~arr[byte_i]);   // 1-bits = prime candidates
        while (primes) {
            int bit_j  = __builtin_ctz((unsigned)primes);
            primes    &= (uint8_t)(primes - 1);      // clear lowest set bit

            uint64_t p = (uint64_t)30 * byte_i + WHEEL30_RESIDUE[bit_j];
            if (UNLIKELY(p < 7)) continue;           // safety: n=1 is marked composite
            if (p > q) return;

            mark_multiples_wheel30(arr, byteCount, p);
        }
    }
}

static size_t prime_sieve_count(const prime_sieve* ps)
{
    size_t         count     = 3;   // primes 2, 3, 5 not stored in array
    const uint8_t* arr       = ps->bits.array;
    const uint64_t limit     = ps->bits.logicalSize;
    const size_t   last_byte = (size_t)(limit / 30);
    const uint32_t lim_mod   = (uint32_t)(limit % 30);
    const size_t   byte_limit = (last_byte < ps->bits.byteSize) ? last_byte : ps->bits.byteSize;

    // Count all 8 candidates in complete bytes
    size_t i = 0;
#if defined(PRIMEC_VECTOR_AVX2)
    // AVX2 VPSHUFB nibble-LUT popcount: 32 bytes per iteration (~32x fewer iterations)
    const __m256i lo_mask  = _mm256_set1_epi8(0x0F);
    const __m256i lut      = _mm256_set_epi8(
        4,3,3,2, 3,2,2,1, 3,2,2,1, 2,1,1,0,
        4,3,3,2, 3,2,2,1, 3,2,2,1, 2,1,1,0);
    const __m256i zero     = _mm256_setzero_si256();
    const __m256i all_ones = _mm256_set1_epi8((char)-1);
    __m256i acc0 = zero, acc1 = zero;
    for (; i + 64 <= byte_limit; i += 64) {
        __m256i v0 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i*)(arr + i     )), all_ones);
        __m256i v1 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i*)(arr + i + 32)), all_ones);
        __m256i cnt0 = _mm256_add_epi8(_mm256_shuffle_epi8(lut, _mm256_and_si256(v0, lo_mask)),
                                       _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(v0, 4), lo_mask)));
        __m256i cnt1 = _mm256_add_epi8(_mm256_shuffle_epi8(lut, _mm256_and_si256(v1, lo_mask)),
                                       _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(v1, 4), lo_mask)));
        acc0 = _mm256_add_epi64(acc0, _mm256_sad_epu8(cnt0, zero));
        acc1 = _mm256_add_epi64(acc1, _mm256_sad_epu8(cnt1, zero));
    }
    for (; i + 32 <= byte_limit; i += 32) {
        __m256i v   = _mm256_xor_si256(_mm256_loadu_si256((const __m256i*)(arr + i)), all_ones);
        __m256i cnt = _mm256_add_epi8(_mm256_shuffle_epi8(lut, _mm256_and_si256(v, lo_mask)),
                                      _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(v, 4), lo_mask)));
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

    // Partial last byte: only residues <= limit % 30
    if (last_byte < ps->bits.byteSize) {
        uint8_t b = (uint8_t)(~arr[last_byte]);
        for (int j = 0; j < 8; j++)
            if (WHEEL30_RESIDUE[j] <= lim_mod && (b & (uint8_t)(1u << j)))
                count++;
    }

    return count;
}

static const struct { uint64_t limit; int count; } results_table[] = {
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

static bool prime_sieve_validate(const prime_sieve* ps)
{
    size_t n = sizeof(results_table) / sizeof(results_table[0]);
    for (size_t i = 0; i < n; i++)
        if (results_table[i].limit == ps->bits.logicalSize)
            return results_table[i].count == (int)prime_sieve_count(ps);
    return false;
}

static void prime_sieve_print(const prime_sieve* ps,
                               bool   showResults,
                               double duration,
                               size_t passes,
                               size_t threads)
{
    if (showResults) fprintf(stderr, "2, ");

    size_t count = (ps->bits.logicalSize >= 2) ? 1 : 0;   // include 2
    for (uint64_t n = 3; n <= ps->bits.logicalSize; n += 2)
        if (bitarray_get(&ps->bits, (size_t)n)) {
            if (showResults) fprintf(stderr, "%llu, ", (unsigned long long)n);
            count++;
        }
    if (showResults) fprintf(stderr, "\n");

    fprintf(stderr,
        "Passes: %zu, Threads: %zu, Time: %g, Average: %g, "
        "Limit: %zu, Counts: %zu/%zu, Valid: %s\n",
        passes, threads, duration, duration / passes,
        ps->bits.logicalSize, count, prime_sieve_count(ps),
        prime_sieve_validate(ps) ? "Pass" : "FAIL!");
    fprintf(stderr, "\n");

    printf("davepl_c_array_optimized;%zu;%g;%zu;algorithm=base,faithful=yes,bits=1\n",
           passes, duration, threads);
}

// ---------------------------------------------------------------------------
// custom_atoll - like atoll but accepts K/M/G/T magnitude suffixes
// ---------------------------------------------------------------------------
static long long custom_atoll(const char* value_str)
{
    char input[256];
    strncpy(input, value_str, sizeof(input) - 1);
    input[sizeof(input) - 1] = '\0';

    for (size_t i = 0; input[i]; i++)
        input[i] = (char)toupper((unsigned char)input[i]);

    size_t len = strlen(input);
    if (len == 0) return 0LL;

    char      last       = input[len - 1];
    long long multiplier = 1LL;
    if      (last == 'K') multiplier = 1000LL;
    else if (last == 'M') multiplier = 1000000LL;
    else if (last == 'G') multiplier = 1000000000LL;
    else if (last == 'T') multiplier = 1000000000000LL;

    if (multiplier != 1LL) {
        input[len - 1] = '\0';
        return (long long)(strtod(input, NULL) * multiplier);
    }
    return strtoll(input, NULL, 10);
}

// ---------------------------------------------------------------------------
// Threading
// ---------------------------------------------------------------------------
typedef struct {
    unsigned int   threadIndex;
    uint64_t*      passes;
    atomic_int*    keepRunning;
    uint64_t       upperLimit;
} ThreadArgs;

static void* thread_func(void* arg)
{
    ThreadArgs* a      = (ThreadArgs*)arg;
    uint64_t    passes = 0;
    prime_sieve sieve;
    prime_sieve_init(&sieve, a->upperLimit);       // allocate once per thread
    while (atomic_load_explicit(a->keepRunning, memory_order_relaxed)) {
        bitarray_reinit(&sieve.bits);              // just memset — no malloc/free
        prime_sieve_run(&sieve);
        passes++;
    }
    prime_sieve_destroy(&sieve);
    *a->passes = passes;
    return NULL;
}

// ---------------------------------------------------------------------------
// Timing helper
// ---------------------------------------------------------------------------
static double timespec_elapsed(const struct timespec* start, const struct timespec* end)
{
    return (double)(end->tv_sec - start->tv_sec)
         + (double)(end->tv_nsec - start->tv_nsec) * 1e-9;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    uint64_t ullLimitRequested = 0;
    int      cThreadsRequested = 0;
    int      cSecondsRequested = 0;
    bool     bPrintPrimes      = false;
    bool     bOneshot          = false;
    bool     bQuiet            = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Syntax: %s [-t,--threads N] [-s,--seconds N] [-l,--limit N] "
                   "[-1,--oneshot] [-p,--print] [-q,--quiet] [-h]\n", argv[0]);
            return 0;
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--threads") == 0) {
            if (++i < argc) cThreadsRequested = atoi(argv[i]);
            if (cThreadsRequested < 1) cThreadsRequested = 1;
        } else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--seconds") == 0) {
            if (++i < argc) cSecondsRequested = atoi(argv[i]);
            if (cSecondsRequested < 1) cSecondsRequested = 1;
        } else if (strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--limit") == 0) {
            if (++i < argc) {
                long long v = custom_atoll(argv[i]);
                ullLimitRequested = (v >= 1) ? (uint64_t)v : 0;
            }
        } else if (strcmp(argv[i], "-1") == 0 || strcmp(argv[i], "--oneshot") == 0) {
            bOneshot          = true;
            cThreadsRequested = 1;
        } else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--print") == 0) {
            bPrintPrimes = true;
        } else if (strcmp(argv[i], "-q") == 0 || strcmp(argv[i], "--quiet") == 0) {
            bQuiet = true;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            return 0;
        }
    }

    if (!bQuiet) {
        fprintf(stderr, "Primes Benchmark (c) 2025 Dave's Garage - https://github.com/davepl/primes\n");
        fprintf(stderr, "--------------------------------------------------------------------------\n");
    }

    if (bOneshot)
        fprintf(stderr, "Oneshot is on. A single pass will be used to simulate a 5 second run.\n");

    if (bOneshot && (cSecondsRequested > 0 || cThreadsRequested > 1)) {
        fprintf(stderr, "Oneshot option cannot be mixed with second count or thread count.\n");
        return 0;
    }

    int      cPasses      = 0;
    int      cSeconds     = cSecondsRequested ? cSecondsRequested : 5;
    int      cThreads     = cThreadsRequested ? cThreadsRequested : (int)sysconf(_SC_NPROCESSORS_ONLN);
    uint64_t llUpperLimit = ullLimitRequested  ? ullLimitRequested  : DEFAULT_UPPER_LIMIT;

    if (!bQuiet) {
        fprintf(stderr, "Computing primes to %llu on %d thread%s for %d second%s.\n",
                (unsigned long long)llUpperLimit,
                cThreads, cThreads == 1 ? "" : "s",
                cSeconds, cSeconds == 1 ? "" : "s");
    }

    double         duration = 0.0;
    struct timespec tStart, tEnd;

    if (bOneshot) {
        clock_gettime(CLOCK_MONOTONIC, &tStart);
        prime_sieve sieve;
        prime_sieve_init(&sieve, llUpperLimit);
        prime_sieve_run(&sieve);
        prime_sieve_destroy(&sieve);
        clock_gettime(CLOCK_MONOTONIC, &tEnd);
        duration = timespec_elapsed(&tStart, &tEnd);
    } else {
        pthread_t*  threads  = (pthread_t*)  malloc((size_t)cThreads * sizeof(pthread_t));
        uint64_t*   l_passes = (uint64_t*)   calloc((size_t)cThreads,  sizeof(uint64_t));
        ThreadArgs* args     = (ThreadArgs*) malloc((size_t)cThreads * sizeof(ThreadArgs));

        atomic_int keepRunning;
        atomic_init(&keepRunning, 1);

        clock_gettime(CLOCK_MONOTONIC, &tStart);

        for (int i = 0; i < cThreads; i++) {
            args[i].threadIndex = (unsigned int)i;
            args[i].passes      = &l_passes[i];
            args[i].keepRunning = &keepRunning;
            args[i].upperLimit  = llUpperLimit;
            pthread_create(&threads[i], NULL, thread_func, &args[i]);
        }

        // Sleep until approximately tStart + cSeconds
        struct timespec tNow;
        clock_gettime(CLOCK_MONOTONIC, &tNow);
        long long elapsed_ns = (long long)(tNow.tv_sec  - tStart.tv_sec)  * 1000000000LL
                             + (long long)(tNow.tv_nsec - tStart.tv_nsec);
        long long remain_ns  = (long long)cSeconds * 1000000000LL - elapsed_ns;
        if (remain_ns > 0) {
            struct timespec rem = { remain_ns / 1000000000LL, remain_ns % 1000000000LL };
            nanosleep(&rem, NULL);
        }

        atomic_store(&keepRunning, 0);
        for (int i = 0; i < cThreads; i++) {
            pthread_join(threads[i], NULL);
            cPasses += (int)l_passes[i];
        }

        clock_gettime(CLOCK_MONOTONIC, &tEnd);
        duration = timespec_elapsed(&tStart, &tEnd);

        free(threads);
        free(l_passes);
        free(args);
    }

    if (bOneshot) {
        cPasses  = (int)(1.0 / duration * 5);
        duration = 5.0;
    }

    prime_sieve checkSieve;
    prime_sieve_init(&checkSieve, llUpperLimit);
    prime_sieve_run(&checkSieve);
    size_t result = prime_sieve_validate(&checkSieve)
                  ? prime_sieve_count(&checkSieve)
                  : 0;

    if (!bQuiet)
        prime_sieve_print(&checkSieve, bPrintPrimes, duration, (size_t)cPasses, (size_t)cThreads);
    else
        fprintf(stderr, "%d, %g\n", cPasses, duration / cPasses);

    prime_sieve_destroy(&checkSieve);

    return (int)result;  // prime count mod 256 as exit code
}
