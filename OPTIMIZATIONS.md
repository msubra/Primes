# PrimeC_array.c — Optimization Log

Platform: Intel Core i9-9980HK (Coffee Lake, 8-core/16-thread, AVX2, ~4.5 GHz turbo)
Sieve limit: N = 1,000,000 | Byte array: ~33,333 bytes (N/30, Wheel-30 encoded)

---

## Baseline Architecture

**Wheel-30 bit-packed sieve**: 1 byte represents 30 consecutive integers, storing 8
residue bits for numbers coprime to 30 (residues {1,7,11,13,17,19,23,29}).
Byte count = N/30 ≈ 33,333 bytes — fits entirely in L1 cache (32 KB on Coffee Lake).

For each prime p ≥ 7, `mark_multiples_wheel30` marks all composites p·q (q ≥ p,
q coprime to 30) using 8 independent strided byte-write loops (one per wheel residue).

---

## Performance Scorecard

| Optimization                        | 1T passes/s | 16T passes/s |
|-------------------------------------|-------------|--------------|
| Baseline (Wheel-30, 8-deep scalar)  | 36,234      | ~201,867     |
| + AVX2 period tile (bigtile)        | 56,509      | 342,329      |
| + Extended_tile fill reduction      | 63,233      | ~350,000     |
| + SIMD cleanup for tail             | **~68,762** | **~360,000** |

---

## Applied Optimizations

### 1. 8-deep scalar unrolling (baseline)

For primes p > 61 (large-prime scalar path), 8 independent load-OR-store byte
operations are issued per outer loop iteration. This keeps 8 independent addresses
in flight for the out-of-order execution engine, hiding address-computation latency.

```c
arr[pos]            |= mask;
arr[pos + stride]   |= mask;
// ... × 8 total
pos += stride * 8;
```

The 8 base addresses (`arr + stride*N` for N=0..7) are hoisted out of the inner
loop by the compiler as loop-invariant, so no multiply runs per iteration.

---

### 2. AVX2 period tile — bigtile (large speedup)

**Key insight**: For a prime p with stride = p bytes, the OR-mask pattern across the
sieve array repeats every p bytes (one full tile period). Instead of 8 separate
strided passes, build a single p-byte combined tile (OR-ing all 8 residue masks),
then apply it using 256-bit AVX2 stores — P stores per outer iteration covering
P×32 bytes.

- Build `tile[p]`: one byte per position 0..p-1, combining all active residue masks.
- Build `bigtile[p × 32]`: tile repeated 32 times so each of the p 32-byte vectors
  can be loaded by index.
- `switch(stride)` with compile-time P lets Clang fully unroll the inner for-loop,
  eliminating branch overhead and enabling register-level loads of each nvec.

**Why AVX2 wins**: Coffee Lake store port (P4) executes 1 store/cycle regardless of
width. A 256-bit store covers 32× more bytes than a byte store for the same cost.
For p=7 (most frequent): ~149 outer iterations vs ~3,984 byte stores — 27× fewer
store-port cycles.

Threshold: stride ≤ 61 (primes 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
59, 61 — 15 primes total).

---

### 3. Extended_tile fill reduction (+12% at 1T)

**Problem**: The bigtile `[p × 32]` fill loop ran p×32 iterations (up to 1,952 for
p=61). This was repeated for every prime every sieve pass.

**Fix**: Replace `bigtile[p×32]` with `extended_tile[p+31]` — just p+31 bytes. For
p=61: 92 fill iterations instead of 1,952 — a **21× reduction**.

The trick: to load the i-th 32-byte AVX2 vector (which starts at tile offset
(i×32) % p), load 32 bytes from `extended_tile[(i×32) % p]`. Since
`extended_tile[0..p+30]` = tile repeated, any 32-byte window starting within
[0, p-1] is valid without wrapping.

The (i×32) % p offset is tracked with a rolling step (no division):
```c
const size_t step = 32u % stride;   // e.g. step=4 for stride=7
size_t off = 0;
for (size_t i = 0; i < stride; i++) {
    nvecs[i] = _mm256_loadu_si256(extended_tile + off);
    off += step;
    if (off >= stride) off -= stride;   // single subtract, no modulo
}
```

---

### 4. SIMD cleanup for the post-tile tail (+7.5% at 1T)

**Problem**: After the main AVX2 tile loop (which processes multiples of p×32 bytes),
a scalar byte-by-byte loop handled the remaining tail. For p=61 this tail can be up
to 61×32-1 = 1,951 bytes — processed one byte at a time.

**Fix**: Reuse the pre-built `nvecs` array in a second SIMD loop for the tail,
processing 32 bytes per step down to the last <32 bytes (handled scalar).

```c
size_t ni = 0;   // nvec index (starts at 0: pos aligned to stride×32)
while (pos + 32u <= byteCount) {
    storeu256(arr+pos, or256(loadu256(arr+pos), nvecs[ni]));
    pos += 32;
    if (++ni == stride) ni = 0;
}
// scalar tail (< 32 bytes)
while (pos < byteCount) { arr[pos++] |= tile[tf]; ... }
```

The `ni` index wraps modulo stride (single compare+subtract per iteration).
A separate `tf` (tile-offset) variable is maintained for the scalar tail only.

---

## Rejected / Reverted Experiments

### 16-deep scalar unrolling → regression (-19% at 1T)

Extending the scalar loop from 8-deep to 16-deep was expected to halve loop
overhead. Empirically it was worse. Root cause: with 16 offsets
(`pos+stride*N`, N=0..15), the compiler generates 15 IMUL instructions (all
competing for execution port 1) per outer iteration setup, whereas 8-deep only
needs 7. The extra dependency chain offset the loop-overhead savings.

### Threshold extension to 79 → 1T regression (-13%), 16T gain (+11%)

Adding primes 67, 71, 73, 79 to the AVX2 switch path (4 new fully-unrolled cases)
consistently hurt single-thread performance while improving multi-thread. Likely
cause: the extra case code bloated the function beyond what fits efficiently in the
CPU's µop cache (DSB = 1536 µops), causing the hot small-prime loops to be
re-decoded more frequently from the MITE decoder.

### Software prefetch in scalar loop → regression (-5.5% at 1T)

`__builtin_prefetch(&arr[pos + stride * 16], 1, 1)` in the 8-deep scalar loop
added load-port pressure without benefit. The 33 KB sieve array fits in L1 cache,
so the hardware prefetcher already keeps the data hot. The explicit prefetch just
consumed load-port bandwidth.

### Medium-prime dynamic SIMD path (62 ≤ p ≤ 181) → 1T regression (-36%), 16T gain (+15%)

Applied the extended_tile approach to medium primes using a dynamic (no-switch)
SIMD loop. Store-port analysis predicted a win (SIMD 32-bit stores vs scalar
8-bit), but the 1T result was much worse. Likely cause: the loop reads and writes
ALL 33K bytes per prime instead of only the ~4K bytes that are actually marked,
significantly increasing total L1 traffic. The 16T gain suggests the approach
reduces inter-thread memory bandwidth contention, which matters at 16 threads but
not at 1.

### AVX2 VPSHUFB popcount in prime_sieve_count → no benchmark impact

Replaced scalar popcnt with an AVX2 VPSHUFB nibble-LUT popcount in
`prime_sieve_count`. Zero benchmark effect because `prime_sieve_count` is called
only once per benchmark run (for final validation), never in the hot sieve loop.
The change was kept for correctness but has no performance value.

---

## Key Architecture Notes (Coffee Lake)

- **Store port P4**: 1 store/cycle regardless of width (8-bit or 256-bit AVX2).
  This is the fundamental bottleneck for the sieve. AVX2 32-byte stores are 32×
  more efficient per store-port cycle than byte stores.
- **Load ports P2/P3**: 2 loads/cycle (256-bit capable). Not the bottleneck.
- **µop cache (DSB)**: 1,536 µops. Large fully-unrolled switch cases can overflow
  this, forcing decode through the slower MITE unit.
- **L1 cache**: 32 KB, 8-way. The 33 KB sieve array plus ~2 KB of nvecs stack data
  slightly exceeds L1 capacity, but the sequential access pattern keeps miss rates
  low via hardware prefetching.
