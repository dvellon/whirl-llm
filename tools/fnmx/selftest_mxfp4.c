// FNMX self-test: patched quantize_mxfp4 (weighted exponent search) vs quantize_row_mxfp4_ref.
// Reports, per input, the weighted error (search weights) and the plain MSE of the dequantized rows.
// PASS requires weighted error (patched) <= weighted error (reference) for every row.
#include "ggml.h"
#include "ggml-quants.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static unsigned long long rng = 88172645463325252ULL;
static float urand(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (float)((rng >> 11) * (1.0 / 9007199254740992.0)); }
static float nrand(void) { float u = urand() + 1e-12f, v = urand(); return sqrtf(-2.0f*logf(u)) * cosf(6.2831853f*v); }

static int run(const char * label, int nrows, int ncols, const float * w, const float * x) {
    size_t rs = ggml_row_size(GGML_TYPE_MXFP4, ncols);
    void * qa = malloc(rs), * qb = malloc(rs);
    float * da = malloc(sizeof(float)*ncols), * db = malloc(sizeof(float)*ncols);
    double wa = 0, wb = 0, ma = 0, mb = 0; int bad = 0, changed_blocks = 0, blocks = 0;
    for (int r = 0; r < nrows; ++r) {
        const float * xr = x + (size_t)r*ncols;
        quantize_row_mxfp4_ref(xr, qa, ncols);
        quantize_mxfp4(xr, qb, 1, ncols, w);
        dequantize_row_mxfp4(qa, da, ncols);
        dequantize_row_mxfp4(qb, db, ncols);
        double s2 = 0; for (int j = 0; j < ncols; ++j) s2 += (double)xr[j]*xr[j]; s2 /= ncols;
        double rwa = 0, rwb = 0;
        for (int j = 0; j < ncols; ++j) {
            double ww = w ? w[j]*sqrt(s2 + (double)xr[j]*xr[j]) : (double)xr[j]*xr[j];
            double ea = da[j]-xr[j], eb = db[j]-xr[j];
            rwa += ww*ea*ea; rwb += ww*eb*eb; ma += ea*ea; mb += eb*eb;
        }
        for (int b = 0; b < ncols/32; ++b) { blocks++; if (((unsigned char*)qa)[b*17] != ((unsigned char*)qb)[b*17]) changed_blocks++; }
        if (rwb > rwa*(1+1e-6) + 1e-30) bad++;
        wa += rwa; wb += rwb;
    }
    printf("%-44s %s rows %4d  weighted err ref %.6e  patched %.6e (%+.2f%%)  MSE ref %.6e patched %.6e (%+.2f%%)  exp changed %.1f%%  rows-worse %d -> %s\n",
        label, w ? "imatrix" : "x^2    ", nrows, wa, wb, 100*(wb/wa-1), ma/nrows/ncols, mb/nrows/ncols, 100*(mb/ma-1), 100.0*changed_blocks/blocks, bad, bad ? "FAIL" : "PASS");
    free(qa); free(qb); free(da); free(db);
    return bad;
}

int main(int argc, char ** argv) {
    int fails = 0;
    // random rows: gaussian and heavy-tailed, weighted and unweighted
    int nr = 256, nc = 2560;
    float * x = malloc(sizeof(float)*nr*nc), * w = malloc(sizeof(float)*nc);
    for (int j = 0; j < nc; ++j) w[j] = 0.1f + 10.0f*urand()*urand();
    for (int i = 0; i < nr*nc; ++i) x[i] = 0.02f*nrand();
    fails += run("random gaussian", nr, nc, NULL, x);
    fails += run("random gaussian", nr, nc, w, x);
    for (int i = 0; i < nr*nc; ++i) { float g = nrand(); x[i] = 0.02f*g*(urand() < 0.02f ? 8.0f : 1.0f); }
    fails += run("random heavy-tail", nr, nc, NULL, x);
    fails += run("random heavy-tail", nr, nc, w, x);
    free(x); free(w);
    // real rows from files: int nrows, int ncols, float w[ncols], float x[nrows*ncols]
    for (int a = 1; a < argc; ++a) {
        FILE * f = fopen(argv[a], "rb"); if (!f) { printf("cannot open %s\n", argv[a]); return 2; }
        int h[2]; fread(h, sizeof(int), 2, f);
        float * ww = malloc(sizeof(float)*h[1]); float * xx = malloc(sizeof(float)*h[0]*h[1]);
        fread(ww, sizeof(float), h[1], f); fread(xx, sizeof(float), (size_t)h[0]*h[1], f); fclose(f);
        const char * base = strrchr(argv[a], 92); base = base ? base+1 : argv[a];
        fails += run(base, h[0], h[1], NULL, xx);
        fails += run(base, h[0], h[1], ww, xx);
        free(ww); free(xx);
    }
    printf("%s\n", fails ? "SELFTEST FAIL" : "SELFTEST PASS");
    return fails ? 1 : 0;
}
