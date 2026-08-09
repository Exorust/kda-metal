/* main.m - parity gate + benchmark for the KDA Metal kernels.
 *
 * One binary, three jobs:
 *   1. parity: chain recurrence steps on CPU reference vs both GPU kernels,
 *      compare state and output within 2^-10 (fp32 vs fp32, expect ~1e-6)
 *   2. determinism: run v2 twice from identical state, require bitwise equality
 *   3. benchmark: GPU-timestamp timing, 2000 chained dispatches per command
 *      buffer so submission cost amortizes out
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TOL 0.0009765625f /* 2^-10 */

/* CPU reference, written from the recurrence contract (decay -> read u from
 * the decayed state -> rank-one delta write -> output from the UPDATED
 * state). fp32 accumulation, like the GPU. */
static void kda_step_ref(float *S, float *o, const float *q, const float *k,
                         const float *v, const float *alpha, float beta,
                         int dk, int dv)
{
    for (int i = 0; i < dk; i++)
        for (int j = 0; j < dv; j++) S[i * dv + j] *= alpha[i];
    for (int j = 0; j < dv; j++) {
        float u = 0.0f;
        for (int i = 0; i < dk; i++) u += k[i] * S[i * dv + j];
        const float d = beta * (v[j] - u);
        float out = 0.0f;
        for (int i = 0; i < dk; i++) {
            S[i * dv + j] += k[i] * d;
            out += q[i] * S[i * dv + j];
        }
        o[j] = out;
    }
}

static unsigned long long s_ = 0x243F6A8885A308D3ull;
static float frnd(void)
{
    s_ = s_ * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((double)(s_ >> 11) / 9007199254740992.0) * 2.0f - 1.0f;
}

static void to_dvmajor(float *dst, const float *src, int H, int dk, int dv)
{
    for (int h = 0; h < H; h++)
        for (int i = 0; i < dk; i++)
            for (int j = 0; j < dv; j++)
                dst[((size_t)h * dv + j) * dk + i] = src[((size_t)h * dk + i) * dv + j];
}
static void from_dvmajor(float *dst, const float *src, int H, int dk, int dv)
{
    for (int h = 0; h < H; h++)
        for (int j = 0; j < dv; j++)
            for (int i = 0; i < dk; i++)
                dst[((size_t)h * dk + i) * dv + j] = src[((size_t)h * dv + j) * dk + i];
}

static id<MTLDevice> g_dev;
static id<MTLCommandQueue> g_q;
static id<MTLComputePipelineState> g_v1, g_v2;
static int npass = 0, nfail = 0;

enum { NDV = 2, NSG = 8 };

static MTLSize grid_v2(int H, int dv)
{ return MTLSizeMake(((size_t)dv + NSG * NDV - 1) / (NSG * NDV), (size_t)H, 1); }

/* one synchronous step dispatch (parity path) */
static void gpu_step(id<MTLComputePipelineState> pso, id<MTLBuffer> __strong *b,
                     int dk, int dv, float qscale, MTLSize grid, MTLSize tg)
{
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [g_q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pso];
        for (int i = 0; i < 7; i++) [e setBuffer:b[i] offset:0 atIndex:i];
        [e setBytes:&dk length:4 atIndex:7];
        [e setBytes:&dv length:4 atIndex:8];
        [e setBytes:&qscale length:4 atIndex:9];
        int ndv = NDV;
        [e setBytes:&ndv length:4 atIndex:10];   /* ignored by v1 */
        [e dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
}

static void check(const char *name, const float *ref, const float *got, size_t n)
{
    float worst = 0.0f, den = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(ref[i] - got[i]);
        if (d > worst) worst = d;
        if (fabsf(ref[i]) > den) den = fabsf(ref[i]);
    }
    float rel = worst / (den > 1e-30f ? den : 1e-30f);
    int ok = rel <= TOL && !isnan(rel);
    printf("  %s  %-22s n=%-7zu rel=%.3e\n", ok ? "PASS" : "FAIL", name, n, rel);
    if (ok) npass++; else nfail++;
}

static void parity(int H, int dk, int dv, int steps)
{
    const size_t sn = (size_t)H * dk * dv;
    float *Sr = malloc(sn * 4);          /* CPU ref, row-major */
    for (size_t i = 0; i < sn; i++) Sr[i] = 0.1f * frnd();
    float *S1 = malloc(sn * 4); memcpy(S1, Sr, sn * 4);   /* v1 */
    float *S2 = malloc(sn * 4); to_dvmajor(S2, Sr, H, dk, dv);
    float *or_ = malloc((size_t)H * dv * 4);
    float *o1 = malloc((size_t)H * dv * 4), *o2 = malloc((size_t)H * dv * 4);
    const float qs = 1.0f;

    id<MTLBuffer> b1S = [g_dev newBufferWithBytes:S1 length:sn * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> b2S = [g_dev newBufferWithBytes:S2 length:sn * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bo = [g_dev newBufferWithLength:(size_t)H * dv * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bq = [g_dev newBufferWithLength:(size_t)H * dk * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bk = [g_dev newBufferWithLength:(size_t)H * dk * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bv = [g_dev newBufferWithLength:(size_t)H * dv * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> ba = [g_dev newBufferWithLength:(size_t)H * dk * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bb = [g_dev newBufferWithLength:(size_t)H * 4 options:MTLResourceStorageModeShared];

    for (int t = 0; t < steps; t++) {
        float *q = bq.contents, *k = bk.contents, *v = bv.contents;
        float *al = ba.contents, *bt = bb.contents;
        for (int i = 0; i < H * dk; i++) { q[i] = frnd(); k[i] = frnd(); }
        for (int i = 0; i < H * dv; i++) v[i] = frnd();
        for (int i = 0; i < H * dk; i++) al[i] = 0.90f + 0.09f * fabsf(frnd());
        for (int h = 0; h < H; h++) bt[h] = 0.5f + 0.4f * fabsf(frnd());

        for (int h = 0; h < H; h++)
            kda_step_ref(Sr + (size_t)h * dk * dv, or_ + (size_t)h * dv,
                         q + (size_t)h * dk, k + (size_t)h * dk,
                         v + (size_t)h * dv, al + (size_t)h * dk, bt[h], dk, dv);

        id<MTLBuffer> v1b[7] = { b1S, bo, bq, bk, bv, ba, bb };
        gpu_step(g_v1, v1b, dk, dv, qs, MTLSizeMake((size_t)H, 1, 1),
                 MTLSizeMake((size_t)dv, 1, 1));
        memcpy(o1, bo.contents, (size_t)H * dv * 4);

        id<MTLBuffer> v2b[7] = { b2S, bo, bq, bk, bv, ba, bb };
        gpu_step(g_v2, v2b, dk, dv, qs, grid_v2(H, dv), MTLSizeMake(32 * NSG, 1, 1));
        memcpy(o2, bo.contents, (size_t)H * dv * 4);
    }

    char lbl[64];
    snprintf(lbl, sizeof lbl, "v1_out H%d D%d", H, dk);   check(lbl, or_, o1, (size_t)H * dv);
    snprintf(lbl, sizeof lbl, "v1_state H%d D%d", H, dk); check(lbl, Sr, b1S.contents, sn);
    snprintf(lbl, sizeof lbl, "v2_out H%d D%d", H, dk);   check(lbl, or_, o2, (size_t)H * dv);
    from_dvmajor(S2, b2S.contents, H, dk, dv);
    snprintf(lbl, sizeof lbl, "v2_state H%d D%d", H, dk); check(lbl, Sr, S2, sn);
    free(Sr); free(S1); free(S2); free(or_); free(o1); free(o2);
}

static double bench_one(id<MTLComputePipelineState> pso, id<MTLBuffer> __strong *b,
                        int dk, int dv, MTLSize grid, MTLSize tg, int steps)
{
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [g_q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pso];
        for (int i = 0; i < 7; i++) [e setBuffer:b[i] offset:0 atIndex:i];
        float qs = 1.0f / sqrtf((float)dk);
        int ndv = NDV;
        [e setBytes:&dk length:4 atIndex:7];
        [e setBytes:&dv length:4 atIndex:8];
        [e setBytes:&qs length:4 atIndex:9];
        [e setBytes:&ndv length:4 atIndex:10];
        for (int s = 0; s < steps; s++)
            [e dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        return (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / steps;
    }
}

static void bench(int H, int dk, int dv, int steps)
{
    const size_t sn = (size_t)H * dk * dv;
    float *init = malloc(sn * 4);
    for (size_t i = 0; i < sn; i++) init[i] = 0.1f * frnd();
    id<MTLBuffer> S = [g_dev newBufferWithBytes:init length:sn * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> o = [g_dev newBufferWithLength:(size_t)H * dv * 4 options:MTLResourceStorageModeShared];
    float *tmp = malloc((size_t)H * dk * 4);
    for (int i = 0; i < H * dk; i++) tmp[i] = 0.01f * (float)(i % 13) - 0.06f;
    id<MTLBuffer> q = [g_dev newBufferWithBytes:tmp length:(size_t)H * dk * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> k = [g_dev newBufferWithBytes:tmp length:(size_t)H * dk * 4 options:MTLResourceStorageModeShared];
    for (int i = 0; i < H * dk; i++) tmp[i] = 0.95f + 0.0004f * (float)(i % 97);
    id<MTLBuffer> a = [g_dev newBufferWithBytes:tmp length:(size_t)H * dk * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> v = [g_dev newBufferWithBytes:tmp length:(size_t)H * dv * 4 options:MTLResourceStorageModeShared];
    float *bb_ = malloc((size_t)H * 4);
    for (int i = 0; i < H; i++) bb_[i] = 0.7f;
    id<MTLBuffer> bt = [g_dev newBufferWithBytes:bb_ length:(size_t)H * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufs[7] = { S, o, q, k, v, a, bt };

    memcpy(S.contents, init, sn * 4);
    double v1 = bench_one(g_v1, bufs, dk, dv, MTLSizeMake((size_t)H, 1, 1),
                          MTLSizeMake((size_t)dv, 1, 1), steps);
    memcpy(S.contents, init, sn * 4);   /* dv-major of a random init: any bytes do */
    double v2 = bench_one(g_v2, bufs, dk, dv, grid_v2(H, dv),
                          MTLSizeMake(32 * NSG, 1, 1), steps);

    /* determinism: identical init, identical steps, bitwise-identical state */
    float *first = malloc(sn * 4);
    memcpy(first, S.contents, sn * 4);
    memcpy(S.contents, init, sn * 4);
    bench_one(g_v2, bufs, dk, dv, grid_v2(H, dv), MTLSizeMake(32 * NSG, 1, 1), steps);
    int bit = memcmp(first, S.contents, sn * 4) == 0;
    if (bit) npass++; else nfail++;

    printf("  H=%-3d D=%-4d %9.2f us %9.2f us  %5.2fx   %s\n",
           H, dk, v1, v2, v1 / v2, bit ? "bit-exact" : "NONDETERMINISTIC");
    free(init); free(tmp); free(bb_); free(first);
}

int main(int argc, char **argv)
{
    @autoreleasepool {
        g_dev = MTLCreateSystemDefaultDevice();
        if (!g_dev) { fprintf(stderr, "no Metal device\n"); return 2; }
        g_q = [g_dev newCommandQueue];
        NSError *err = nil;
        NSString *src = [NSString stringWithContentsOfFile:@(argc > 1 ? argv[1] : "kda.metal")
                                                  encoding:NSUTF8StringEncoding error:&err];
        if (!src) { fprintf(stderr, "cannot read kda.metal\n"); return 2; }
        MTLCompileOptions *opts = [MTLCompileOptions new];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        opts.fastMathEnabled = NO;   /* precise exp/tanh contract */
#pragma clang diagnostic pop
        id<MTLLibrary> lib = [g_dev newLibraryWithSource:src options:opts error:&err];
        if (!lib) { fprintf(stderr, "%s\n", err.localizedDescription.UTF8String); return 2; }
        g_v1 = [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_kda_step"] error:&err];
        g_v2 = [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_kda_step2"] error:&err];
        if (!g_v1 || !g_v2) { fprintf(stderr, "PSO build failed\n"); return 2; }

        printf("== KDA on Metal ==  device: %s\n", g_dev.name.UTF8String);
        printf("-- parity vs CPU reference (4 chained steps, state + output) --\n");
        parity(2, 16, 16, 4);
        parity(4, 32, 32, 4);
        parity(96, 128, 128, 4);
        printf("-- benchmark (GPU time, %d chained dispatches) + v2 determinism --\n", 2000);
        printf("  %-11s %10s %11s %7s\n", "shape", "v1/step", "v2/step", "speed");
        bench(96, 128, 128, 2000);
        bench(48, 128, 128, 2000);
        bench(2, 16, 16, 2000);
        printf("%d passed, %d failed\n", npass, nfail);
        return nfail ? 1 : 0;
    }
}
