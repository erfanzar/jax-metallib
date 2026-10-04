# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MPS lowering for ``lax.rng_bit_generator_p`` (ThreeFry / Philox expanded in uint32 ops).

The native ``stablehlo.rng_bit_generator`` handler ignores the input state, returns a
zero state, rejects uint32 output and aborts the process (SIGABRT) for uint8/uint16.
That made ``lax.rng_bit_generator`` and ``jax.random`` with ``impl='rbg'`` /
``'unsafe_rbg'`` unusable on MPS. This lowering reproduces XLA's
``RngBitGeneratorExpander`` (bits and returned state) with uint32 arithmetic only, so
it needs no x64 and matches XLA:CPU bit for bit. ``RNG_DEFAULT`` is Philox, as on
XLA:CPU. ``JAX_METALLIB_DISABLE_RBG_LOWERING=1`` keeps JAX's default lowering.
"""

from __future__ import annotations

import math
import os

import numpy as np

_U32 = np.uint32


def _disabled() -> bool:
    return os.environ.get("JAX_METALLIB_DISABLE_RBG_LOWERING", "0") not in ("", "0")


def _add64(lo, hi, inc):
    """(hi:lo) + inc as a 64-bit add on uint32 words; inc is a uint32 array or scalar."""
    new_lo = lo + inc
    carry = (new_lo < lo).astype(_U32)
    return new_lo, hi + carry


def _mulhilo(a, b: int):
    """32x32 -> 64-bit product of a uint32 array and a constant, as (high, low) words."""
    m16 = _U32(0xFFFF)
    s16 = _U32(16)
    b0, b1 = _U32(b & 0xFFFF), _U32(b >> 16)
    a0, a1 = a & m16, a >> s16
    p00, p01, p10, p11 = a0 * b0, a0 * b1, a1 * b0, a1 * b1
    mid = (p00 >> s16) + (p01 & m16) + (p10 & m16)
    hi = p11 + (p01 >> s16) + (p10 >> s16) + (mid >> s16)
    lo = (mid << s16) | (p00 & m16)
    return hi, lo


def _philox_cut_disabled() -> bool:
    return os.environ.get("JAX_METALLIB_DISABLE_PHILOX_ROUND_CUT", "0") not in ("", "0")


def _philox4x32(x0, x1, x2, x3, k0, k1, cut=False):
    """Philox-4x32, 10 rounds. With ``cut`` the four mulhilo outputs of every round and the four
    words after each of the first 9 rounds are reinterpreted through int32 (see ``_cut``), which
    keeps MPSGraph's fused pieces short: jax.random impl='rbg' bits 2^20 7.48 -> 4.82 ms, 2^24
    196.1 -> 115.7 ms, bitwise vs XLA:CPU (build/probe_philox_cut*_r19v.log). Cutting only the words
    every 2/5 rounds gave 163.5/189.5 ms; also cutting the partial products inside _mulhilo did not
    help (121.1 ms)."""
    for i in range(10):
        hi0, lo0 = _mulhilo(x0, 0xD2511F53)
        hi1, lo1 = _mulhilo(x2, 0xCD9E8D57)
        if cut:
            hi0, lo0, hi1, lo1 = _cut(hi0), _cut(lo0), _cut(hi1), _cut(lo1)
        x0, x1, x2, x3 = hi1 ^ x1 ^ k0, lo1, hi0 ^ x3 ^ k1, lo0
        k0 = k0 + _U32(0x9E3779B9)
        k1 = k1 + _U32(0xBB67AE85)
        if cut and i < 9:
            x0, x1, x2, x3 = _cut(x0), _cut(x1), _cut(x2), _cut(x3)
    return x0, x1, x2, x3


def _rotl(x, d: int):
    return (x << _U32(d)) | (x >> _U32(32 - d))


def _round_cut_disabled() -> bool:
    return os.environ.get("JAX_METALLIB_DISABLE_THREEFRY_ROUND_CUT", "0") not in ("", "0")


def _cut(x):
    """uint32 -> int32 -> uint32 reinterpret: the identity on bits, but MPSGraph does not fuse across it."""
    from jax import lax

    return lax.bitcast_convert_type(lax.bitcast_convert_type(x, np.int32), _U32)


def _threefry2x32(k0, k1, x0, x1, cut=False):
    """ThreeFry-2x32, 20 rounds. With ``cut`` the two words are reinterpreted through int32 after
    each of the first 19 rounds. The unbroken ~100-op uint32 chain exceeds MPSGraph's stitching
    budget and runs unfused; the cuts give short fused pieces: jax.random.bits 2^20 6.56 -> 4.39 ms,
    2^24 154.9 -> 90.0 ms, bitwise vs XLA:CPU (build/probe_threefry_p_cut*_r19u.log). An f32
    reinterpret was as fast, but int32 cannot touch NaN payloads."""
    ks = (k0, k1, k0 ^ k1 ^ _U32(0x1BD11BDA))
    rotations = ((13, 15, 26, 6), (17, 29, 16, 24))
    x0 = x0 + k0
    x1 = x1 + k1
    done = 0
    for i in range(5):
        for r in rotations[i % 2]:
            x0 = x0 + x1
            x1 = _rotl(x1, r) ^ x0
            done += 1
            if cut and done < 20:
                x0, x1 = _cut(x0), _cut(x1)
        x0 = x0 + ks[(i + 1) % 3]
        x1 = x1 + ks[(i + 2) % 3] + _U32(i + 1)
    return x0, x1


def _threefry2x32_p_impl(k1, k2, x1, x2, *, shape):
    """threefry2x32_p body for MPS: broadcast the operands to the output shape, then hash."""
    import jax.numpy as jnp

    k1, k2, x1, x2 = (jnp.broadcast_to(v, shape) for v in (k1, k2, x1, x2))
    return list(_threefry2x32(k1, k2, x1, x2, cut=True))


def _threefry_split_axis(shape) -> int:
    """Axis XLA:CPU splits for narrow ThreeFry output (measured): the first even dimension,
    else the largest dimension (first on ties), which is then padded by one."""
    for i, s in enumerate(shape):
        if s % 2 == 0:
            return i
    return max(range(len(shape)), key=lambda i: (shape[i], -i))


def _expand(key, *, shape, dtype, algorithm):
    """XLA RngBitGeneratorExpander for a u32[4] key (u64[2] state = [key64, state64])."""
    import jax.numpy as jnp
    from jax import lax

    dtype = np.dtype(dtype)
    n = math.prod(shape)
    if n == 0:
        # No counters are consumed, so the state is unchanged (checked against XLA:CPU). Skipping
        # the generator avoids zero-element intermediates, which crash the MPS executor.
        return [key, jnp.zeros(shape, dtype)]
    # Route the key through a gather before extracting its words. When the key is not a direct
    # graph argument (vmap's keys[0], any computed key), MPSGraph otherwise spent > 60 s compiling
    # the ~250-op ThreeFry graph (MemrefDeallocationPass); after the gather it takes ~0.06 s, and
    # the results stay bitwise (build/probe_threefry_keyvariants2_r19u.log). A masked-reduce key
    # still hung there, so the gather is the barrier, not just any computed value.
    key = jnp.take(key, lax.iota(np.int32, 4), mode="clip")
    k0, k1, s_lo, s_hi = key[0], key[1], key[2], key[3]
    wide = dtype.itemsize == 8

    if algorithm == lax.RandomAlgorithm.RNG_THREE_FRY:
        if wide:
            # 64-bit output: one counter per element (word 0 low, word 1 high).
            count = n
            c_lo, c_hi = _add64(s_lo, s_hi, lax.iota(_U32, count))
            words = _threefry2x32(k0, k1, c_lo, c_hi, cut=not _round_cut_disabled())
        else:
            # Narrower output: split one axis d in halves (padded to even), run one counter per
            # half-shape element in row-major order, and interleave the two words along d:
            # out[..., 2q + w, ...] = word_w(half[..., q, ...]).
            work = tuple(shape) or (1,)
            d = _threefry_split_axis(work)
            padded = list(work)
            padded[d] += padded[d] % 2
            half = list(padded)
            half[d] //= 2
            count = math.prod(half)
            c_lo, c_hi = _add64(s_lo, s_hi, lax.iota(_U32, count))
            o0, o1 = _threefry2x32(k0, k1, c_lo, c_hi, cut=not _round_cut_disabled())
            pair = jnp.stack([o0.reshape(half), o1.reshape(half)], axis=d + 1).reshape(padded)
            words = (lax.slice_in_dim(pair, 0, work[d], axis=d).reshape(-1),)
        new_lo, new_hi = _add64(s_lo, s_hi, _U32(count))
    else:  # RNG_PHILOX and RNG_DEFAULT (Philox on XLA:CPU)
        # A u64[2] state is reversed into the 128-bit counter: low = state64, high = key64.
        # Each counter yields four u32 words: four narrow elements or two u64 elements.
        groups = math.ceil(n / 2) if wide else math.ceil(n / 4)
        c_lo, c_hi = _add64(s_lo, s_hi, lax.iota(_U32, groups))
        overflow = ((c_hi < s_hi) | ((c_hi == s_hi) & (c_lo < s_lo))).astype(_U32)
        h_lo, h_hi = _add64(jnp.broadcast_to(k0, (groups,)), jnp.broadcast_to(k1, (groups,)), overflow)
        o = _philox4x32(c_lo, c_hi, h_lo, h_hi, k0, k1, cut=not _philox_cut_disabled())
        words = None
        if wide:
            lo_w = jnp.stack([o[0], o[2]], axis=1).reshape(-1)[:n]
            hi_w = jnp.stack([o[1], o[3]], axis=1).reshape(-1)[:n]
            words = (lo_w, hi_w)
        else:
            words = (jnp.stack(o, axis=1).reshape(-1)[:n],)
        new_lo, new_hi = _add64(s_lo, s_hi, _U32(groups))

    if wide:
        bits = lax.convert_element_type(words[0], np.uint64) | (
            lax.convert_element_type(words[1], np.uint64) << np.uint64(32)
        )
    elif dtype.itemsize == 4:
        bits = words[0]
    else:
        mask = _U32((1 << (8 * dtype.itemsize)) - 1)
        bits = lax.convert_element_type(words[0] & mask, dtype)
    bits = bits.reshape(shape)
    new_key = jnp.stack([k0, k1, new_lo, new_hi])
    return [new_key, bits]


def _register_threefry_p() -> None:
    """MPS lowering for threefry2x32_p (the default jax.random PRNG) with per-round fusion cuts."""
    from jax._src import core
    from jax._src.interpreters import mlir
    from jax._src.random import threefry2x32 as tf

    def _rule(ctx, k1, k2, x1, x2):
        aval = ctx.avals_out[0]
        if not core.is_constant_shape(aval.shape):
            return tf._threefry2x32_lowering_rule(ctx, k1, k2, x1, x2)
        if 0 in aval.shape:
            # As JAX's GPU rule: no hashing for empty output (zero-element intermediates crash MPS).
            zeros = mlir.full_like_aval(ctx, 0, aval)
            return [zeros, zeros]
        body = mlir.lower_fun(lambda *a: _threefry2x32_p_impl(*a, shape=tuple(aval.shape)), multiple_results=True)
        return body(ctx, k1, k2, x1, x2)

    mlir.register_lowering(tf.threefry2x32_p, _rule, platform="mps")


def register() -> None:
    """Register the MPS random lowerings. Call after the 'mps' plugin is registered."""
    if not _round_cut_disabled():
        _register_threefry_p()
    if _disabled():
        return
    from jax._src import core
    from jax._src.interpreters import mlir
    from jax._src.lax import lax as lax_internal

    expand_rule = mlir.lower_fun(_expand, multiple_results=True)

    def _rbg_rule(ctx, key, *, shape, dtype, algorithm, out_sharding):
        key_aval = ctx.avals_in[0]
        supported = (
            key_aval.dtype == np.dtype(np.uint32)
            and tuple(key_aval.shape) == (4,)
            and all(core.is_constant_shape(a.shape) for a in ctx.avals_out)
            and math.prod(shape) < 2**32
        )
        if not supported:
            return lax_internal._rng_bit_generator_lowering(
                ctx, key, shape=shape, dtype=dtype, algorithm=algorithm, out_sharding=out_sharding
            )
        return expand_rule(ctx, key, shape=shape, dtype=dtype, algorithm=algorithm)

    mlir.register_lowering(lax_internal.rng_bit_generator_p, _rbg_rule, platform="mps")
