/*
 * Shared xoshiro128++ PRNG with SplitMix64 seeding for brileta native code.
 *
 * Include this header in any C source file that needs a fast, reproducible
 * PRNG.  All functions are static inline so each translation unit gets its
 * own copy without linker conflicts.
 */

#ifndef BRILETA_NATIVE_RNG_H
#define BRILETA_NATIVE_RNG_H

#include <stdint.h>

/* ------------------------------------------------------------------ */
/* xoshiro128++ with SplitMix64 seeding                                */
/* ------------------------------------------------------------------ */

/*
 * NativeRng is the local PRNG state used by native solvers and sprite
 * generators.
 *
 * Why xoshiro128++:
 * - Fast enough for tight inner loops.
 * - Good statistical quality for game/procedural content.
 * - Small state footprint (4x32-bit).
 *
 * We seed xoshiro with SplitMix64 because:
 * - Python provides a single 64-bit seed.
 * - xoshiro needs multiple non-zero state words.
 * - SplitMix64 expands one seed into well-scrambled state values.
 */
typedef struct {
    uint32_t s[4];
} NativeRng;

static inline uint32_t native_rotl32(uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

static inline uint64_t native_splitmix64_next(uint64_t *state) {
    uint64_t z;

    *state += 0x9E3779B97F4A7C15ULL;
    z = *state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static inline void native_rng_init(NativeRng *rng, uint64_t seed) {
    uint64_t sm = seed;
    uint64_t a = native_splitmix64_next(&sm);
    uint64_t b = native_splitmix64_next(&sm);

    rng->s[0] = (uint32_t)a;
    rng->s[1] = (uint32_t)(a >> 32);
    rng->s[2] = (uint32_t)b;
    rng->s[3] = (uint32_t)(b >> 32);

    /* xoshiro cannot run with an all-zero state. */
    if ((rng->s[0] | rng->s[1] | rng->s[2] | rng->s[3]) == 0) {
        rng->s[0] = 0x9E3779B9U;
        rng->s[1] = 0x243F6A88U;
        rng->s[2] = 0xB7E15162U;
        rng->s[3] = 0x8AED2A6BU;
    }
}

static inline uint32_t native_rng_next_u32(NativeRng *rng) {
    /* xoshiro128++ output scrambler. */
    uint32_t result = native_rotl32(rng->s[0] + rng->s[3], 7) + rng->s[0];
    uint32_t t = rng->s[1] << 9;

    rng->s[2] ^= rng->s[0];
    rng->s[3] ^= rng->s[1];
    rng->s[1] ^= rng->s[2];
    rng->s[0] ^= rng->s[3];

    rng->s[2] ^= t;
    rng->s[3] = native_rotl32(rng->s[3], 11);

    return result;
}

/*
 * Generate a double in [0, 1) from the upper 53 random bits.
 * This mirrors common high-quality float conversion schemes and avoids
 * leaning on lower bits, which are the weakest bits for xoshiro/xoroshiro
 * family generators.
 */
static inline double native_rng_next_double(NativeRng *rng) {
    uint64_t hi = (uint64_t)(native_rng_next_u32(rng) >> 5);     /* 27 bits */
    uint64_t lo = (uint64_t)(native_rng_next_u32(rng) >> 6);     /* 26 bits */
    uint64_t mantissa = (hi << 26) | lo;                          /* 53 bits */
    return (double)mantissa * (1.0 / 9007199254740992.0);         /* 2^-53   */
}

#endif /* BRILETA_NATIVE_RNG_H */
