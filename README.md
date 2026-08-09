# KDA on Metal

**Kimi Delta Attention — the gated delta-rule linear attention behind Kimi K3 —
as a pair of Apple Metal kernels: the straightforward implementation and a
register-resident rewrite that is ~2x faster, with parity gates against a CPU
reference and bit-exact run-to-run determinism.**

Three files: `kda.metal` (both kernels), `main.m` (CPU reference + parity +
determinism + benchmark), `Makefile`.

```
make test
```

## The recurrence

69 of Kimi K3's 93 layers use KDA instead of softmax attention. Each head
carries a 128x128 state matrix `S`, updated per token — order is load-bearing:

```
1. decay   S[i][:] *= alpha[i]        channel-wise forget gate
2. read    u = S^T k                  what the state predicts for this key
3. write   S += k (beta (v - u))^T    the prediction ERROR, rank one
4. output  o = S^T q                  from the ALREADY-UPDATED state
```

Step 3 makes it a *delta* rule — the state absorbs only what it got wrong.
No KV cache; O(1) memory per layer at any context length.
(Algorithm: Kimi Linear / `fla` KDA. Engine that motivated this:
[kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c); full Metal
backend at [kimi-k3-metal](https://github.com/Exorust/kimi-k3-metal).)

## v1 vs v2

**v1 (`k_kda_step`)** — one *thread* per state column, row-major state, scalar
loads. Correct and deterministic, but per thread it issues ~256 scalar state
loads + 128 stores, plus ~768 redundant broadcast loads (all 128 threads
re-read all of `k`, `alpha`, `q`), and touches the state three times per token.

**v2 (`k_kda_step2`)** — the register-resident shape:

- state transposed to **dv-major** `[head][j][i]`: one **simdgroup** per
  column, lane `L` holds `S[j][4L..4L+3]` as a `float4` **in registers**
- `k`, `alpha`, `q` loaded **once per simdgroup** as `float4`, reused across
  its columns
- decayed column computed in registers → state touched **once read, once
  write** (2x traffic instead of 3x)
- `u` and `o` via `simd_sum` — a fixed shuffle tree, so results are
  deterministic; columns are independent, so there are **zero barriers**

Per lane, ~1,150 memory instructions become ~19 vector operations.

## Measured (Apple M5, GPU timestamps, 2000 chained dispatches/cb)

| shape (one KDA layer) | v1 / step | v2 / step | speedup | run-to-run |
|---|---|---|---|---|
| real: H=96, D=128 (6 MB state) | 88.02 us | 44.00 us | **2.00x** | bit-exact |
| H=48, D=128 | 44.43 us | 21.86 us | **2.03x** | bit-exact |
| tiny: H=2, D=16 | 3.64 us | 1.29 us | **2.83x** | bit-exact |

At Kimi K3 scale (69 KDA layers) v2 returns **~3 ms of GPU time per token**.

Numbers are from `make test` on this repo — reproduce them yourself. Parity:
both kernels track a CPU reference through chained steps within ~6e-7 relative
(gate: 2^-10), state and outputs both compared. Determinism: two runs from
identical state are `memcmp`-equal.

Honest footnotes: the 6 MB working set is partly cache-resident, so treat the
*ratio* as the robust number, and this isolates kernel GPU time — in a full
engine the win also requires batched command encoding around it.

## Constraints both kernels honor

fp32 accumulation, fast-math OFF (precise transcendentals), no atomics, no
data-dependent reduction order — bit-exact reproducibility by construction,
which is what lets an inference engine gate a GPU backend against a CPU
reference.
