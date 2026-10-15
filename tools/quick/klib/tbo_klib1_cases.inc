    // KLIB-1 (WHIRL tier-0, local patch, not upstream): Qwen3.8-27B shapes only, when KLIB1=1
    if (getenv("KLIB1")) {
        const ggml_type qt[] = {GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_IQ4_XS, GGML_TYPE_Q6_K};
        for (ggml_type t : qt) {
            for (int n : {1, 4}) {
                test_cases.emplace_back(new test_mul_mat(t, GGML_TYPE_F32, 17408, n, 5120, {1, 1}, {1, 1}));
                test_cases.emplace_back(new test_mul_mat(t, GGML_TYPE_F32, 5120, n, 17408, {1, 1}, {1, 1}));
            }
        }
        // prefill GEMM, 2k chunk (ggml: MMQ = q8_1 activations for quantized A; F16 A = f16 GEMM)
        for (ggml_type t : {GGML_TYPE_Q4_K, GGML_TYPE_IQ4_XS, GGML_TYPE_F16}) {
            test_cases.emplace_back(new test_mul_mat(t, GGML_TYPE_F32, 17408, 2048, 5120, {1, 1}, {1, 1}));
            test_cases.emplace_back(new test_mul_mat(t, GGML_TYPE_F32, 5120, 2048, 17408, {1, 1}, {1, 1}));
        }
        // gated DeltaNet prefill chunk: 16 k heads x 3 (48 v heads), dk = dv = 128
        for (int nt : {2048, 1}) test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 16, 128, nt, 1, 3));
        // flash attention, 24 / 4 heads, hd 256, 2048 queries, f16 KV (mask = random causal-like from the test)
        for (int kv : {4096, 32768, 131072})
            test_cases.emplace_back(new test_flash_attn_ext(256, 256, 4, {6, 1}, kv, 2048, true, false, 0.0f, 0.0f, GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16));
        return test_cases;
    }
