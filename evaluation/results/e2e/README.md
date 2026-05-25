# End-to-End Pipeline Results

End-to-end evaluations where the retriever is live (not gold tables). Uses Gemma 4 31B.

The `e2e_*` files are earlier ablation runs testing combinations of SKOS resolver,
QUDT unit validation, and the C5 execution loop with the full retrieval pipeline.
`smoke` files are small sanity-check subsets.

`vllm_gemma4_e2e_baseline_*` is the full baseline end-to-end run (no C5 enhancements).
