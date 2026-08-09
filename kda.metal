/* kda.metal - Kimi Delta Attention (gated delta rule) on Apple Metal.
 *
 * The recurrence, per head, per token (order is load-bearing):
 *   1. decay   S[i][:] *= alpha[i]     channel-wise forget gate
 *   2. read    u = S^T k               what the state predicts for this key
 *   3. write   S += k (beta (v-u))^T   the prediction ERROR, rank one
 *   4. output  o = S^T q               from the ALREADY-UPDATED state
 * q arrives pre-scaled by d_k^-0.5 (or fold via qscale).
 * Algorithm: Kimi Linear / fla KDA; engine reference: kimi-k3-in-c.
 *
 * Both kernels: fp32 accumulate, no atomics, no data-dependent reduction
 * order -- bit-exact run-to-run by construction. Compile with fast math OFF.
 */
#include <metal_stdlib>
using namespace metal;

/* One threadgroup per head, one thread per COLUMN j of the per-head state
 * S[dk][dv] (row-major, the CPU's layout: at fixed row i, threads j read
 * consecutive addresses -- coalesced, no transpose needed). Each column is
 * fully independent, so the load-bearing order (decay, read u from the
 * decayed state, delta-write, output from the UPDATED state, k3.h:437-443)
 * runs without a single barrier. State lives in device memory: D*D*4 bytes
 * per head (64KB at D=128) cannot sit in the 32KB threadgroup space
 * (DESIGN.md finding 4). q arrives unscaled; qscale folds the CPU's d_k^-0.5
 * pre-scaling (pass 1.0 to match bare k3_kda_step). */
kernel void k_kda_step(device float       *S     [[buffer(0)]],
                       device float       *o     [[buffer(1)]],
                       device const float *q     [[buffer(2)]],
                       device const float *k     [[buffer(3)]],
                       device const float *v     [[buffer(4)]],
                       device const float *alpha [[buffer(5)]],
                       device const float *beta  [[buffer(6)]],
                       constant int   &dk     [[buffer(7)]],
                       constant int   &dv     [[buffer(8)]],
                       constant float &qscale [[buffer(9)]],
                       uint h [[threadgroup_position_in_grid]],
                       uint j [[thread_position_in_threadgroup]])
{
    if ((int)j >= dv) return;
    device float       *Sh = S + (size_t)h * dk * dv;
    device const float *qh = q + (size_t)h * dk;
    device const float *kh = k + (size_t)h * dk;
    device const float *ah = alpha + (size_t)h * dk;
    const float vj = v[(size_t)h * dv + j];
    const float bt = beta[h];

    /* u_j = sum_i k_i * (alpha_i * S_ij): the DECAYED state, read-only pass */
    float u = 0.0f;
    for (int i = 0; i < dk; i++)
        u += kh[i] * (ah[i] * Sh[(size_t)i * dv + j]);

    /* decay + rank-one delta write + output from the updated state */
    const float dlt = bt * (vj - u);
    float out = 0.0f;
    for (int i = 0; i < dk; i++) {
        const float s = ah[i] * Sh[(size_t)i * dv + j] + kh[i] * dlt;
        Sh[(size_t)i * dv + j] = s;
        out += (qh[i] * qscale) * s;
    }
    o[(size_t)h * dv + j] = out;
}

/* The optimized shape (PLAN.md M6; prior art: mtplx gdn_linear_attention).
 * State is DV-MAJOR on the GPU: S2[head][j][i], dk contiguous, so one
 * simdgroup owns column j and lane L holds S[j][4L..4L+3] as a float4 IN
 * REGISTERS. k/alpha/q are loaded once per simdgroup as float4 and reused
 * across its ndv columns. One state read + one state write per token
 * (v1 does two reads + one write, all scalar, and re-reads k/alpha/q from
 * device memory dk times per column).
 * Order is the k3.h contract exactly: decay -> u from the DECAYED state ->
 * delta-write -> output from the UPDATED state. simd_sum is a fixed shuffle
 * tree: deterministic, no atomics, fp32 throughout.
 * Host guarantees dk % 4 == 0 and dk <= 128. */
kernel void k_kda_step2(device float       *S     [[buffer(0)]],  /* [H][dv][dk] */
                        device float       *o     [[buffer(1)]],
                        device const float *q     [[buffer(2)]],
                        device const float *k     [[buffer(3)]],
                        device const float *v     [[buffer(4)]],
                        device const float *alpha [[buffer(5)]],
                        device const float *beta  [[buffer(6)]],
                        constant int   &dk     [[buffer(7)]],
                        constant int   &dv     [[buffer(8)]],
                        constant float &qscale [[buffer(9)]],
                        constant int   &ndv    [[buffer(10)]],
                        uint2 tg   [[threadgroup_position_in_grid]],
                        uint  sgid [[simdgroup_index_in_threadgroup]],
                        uint  nsg  [[simdgroups_per_threadgroup]],
                        uint  lane [[thread_index_in_simdgroup]])
{
    const uint h = tg.y;
    device float       *Sh = S + (size_t)h * dk * dv;
    device const float *qh = q + (size_t)h * dk;
    device const float *kh = k + (size_t)h * dk;
    device const float *ah = alpha + (size_t)h * dk;
    const float bt = beta[h];

    const int  e0  = 4 * (int)lane;
    const bool act = e0 < dk;
    float4 k4 = 0.0f, a4 = 0.0f, q4 = 0.0f;
    if (act) {
        k4 = *(device const float4 *)(kh + e0);
        a4 = *(device const float4 *)(ah + e0);
        q4 = *(device const float4 *)(qh + e0) * qscale;
    }

    const int col0 = (int)(tg.x * nsg + sgid) * ndv;
    for (int cc = 0; cc < ndv; cc++) {
        const int j = col0 + cc;
        if (j >= dv) break;
        device float *col = Sh + (size_t)j * dk;

        float4 s4 = act ? *(device const float4 *)(col + e0) : float4(0.0f);
        const float4 ds = a4 * s4;                    /* decayed, in registers */
        float u = simd_sum(k4.x * ds.x + k4.y * ds.y + k4.z * ds.z + k4.w * ds.w);
        const float dlt = bt * (v[(size_t)h * dv + j] - u);
        const float4 ns = ds + k4 * dlt;              /* delta write */
        if (act) *(device float4 *)(col + e0) = ns;
        const float op =
            simd_sum(q4.x * ns.x + q4.y * ns.y + q4.z * ns.z + q4.w * ns.w);
        if (lane == 0) o[(size_t)h * dv + j] = op;
    }
}
