# Baseline & Comparison Results

Pre-C5 and comparison model evaluations, query-only (gold tables provided).

| File | Description |
|------|-------------|
| `baseline_out/res.json` | Initial vLLM baseline run |
| `gemma4_baseline_rerun_*` | Gemma 4 31B baseline re-run after code changes |
| `gemma4_31b_colbert_*` | Gemma 4 31B with ColBERT retriever (end-to-end baseline) |
| `kg_resolved_gemma4_*` | C4 SKOS resolver evaluation, query-only |
| `kg_resolved_gemma4_e2e_*` | C4 SKOS resolver evaluation, end-to-end with retriever |
| `vllm_qwen35_*` | Standalone Qwen 3.5 model explorations (not C5) |
