# KDA on Metal

Kimi K3 swaps softmax attention for a gated delta rule in 69 of its 93 layers. This repo is that recurrence on Apple GPUs, written twice: `k_kda_step`, the obvious implementation, and `k_kda_step2`, a register-resident rewrite that runs about 2x faster. Both are bit-exact across runs, and both are gated against a CPU reference.

Three files: `kda.metal` (both kernels), `main.m` (CPU reference, parity, determinism check, benchmark), `Makefile`.

```
make test
```

## The recurrence

Each KDA head carries a 128x128 state matrix `S`, updated once per token. The order matters:

```
1. decay   S[i][:] *= alpha[i]        channel-wise forget gate
2. read    u = S^T k                  what the state predicts for this key
3. write   S += k (beta (v - u))^T    the prediction ERROR, rank one
4. output  o = S^T q                  from the ALREADY-UPDATED state
```

Step 3 is the delta rule: the state absorbs only what it got wrong. There is no KV cache, so memory per layer stays flat at any context length. `q` arrives pre-scaled by `d_k^-0.5`, or you fold it in via `qscale`.

The algorithm comes from Kimi Linear (`fla` KDA). The engine that motivated this port is [kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c); the full Metal backend lives at [kimi-k3-metal](https://github.com/Exorust/kimi-k3-metal).

## Two kernels

v1 is one thread per state column: row-major state, scalar loads. Correct, deterministic, and slow in a specific way. Each thread issues about 256 scalar state loads plus 128 stores, all 128 threads re-read the whole of `k`, `alpha`, and `q` (768 redundant loads per thread), and the state gets touched three times per token.

v2 keeps the math and changes where the data lives:

- The state is transposed to dv-major `[head][j][i]`. One simdgroup owns column `j`, and lane `L` holds `S[j][4L..4L+3]` as a `float4` in registers.
- `k`, `alpha`, and `q` are loaded once per simdgroup as `float4` and reused across its columns.
- The decayed column is computed in registers, so the state is read once and written once per token instead of read twice and written once.
- `u` and `o` come out of `simd_sum`, a fixed shuffle tree. Columns are independent, so the kernel contains no barriers.

Per lane, roughly 1,150 memory instructions become about 19 vector operations.

## Numbers

Apple M5, GPU timestamps, 2000 chained dispatches per command buffer:

| shape (one KDA layer) | v1 / step | v2 / step | speedup | run-to-run |
|---|---|---|---|---|
| real: H=96, D=128 (6 MB state) | 88.02 us | 44.00 us | 2.00x | bit-exact |
| H=48, D=128 | 44.43 us | 21.86 us | 2.03x | bit-exact |
| tiny: H=2, D=16 | 3.64 us | 1.29 us | 2.83x | bit-exact |

At Kimi K3 scale, 69 KDA layers, v2 gives back about 3 ms of GPU time per token.

`make test` reproduces the table and the checks behind it. Both kernels track the CPU reference through chained steps within about 6e-7 relative error against a 2^-10 gate, comparing state and outputs at every shape. Two v2 runs from identical state are memcmp-equal.

Two caveats worth stating plainly: the 6 MB working set is partly cache-resident, so trust the ratio, not the GB/s. And this measures kernel GPU time in isolation; inside a full engine the win also needs batched command encoding around it.

## The contract

fp32 accumulation, fast math off, no atomics, no data-dependent reduction order. That is what makes both kernels bit-exact across runs, and bit-exactness is what lets an inference engine gate a GPU backend against a CPU reference instead of hoping.
