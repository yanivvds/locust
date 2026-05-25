# Table Retrieval Results

ColBERT (GECKOv2-EN) table retrieval evaluations on the EN test set (307 questions).

| File | Description |
|------|-------------|
| `table_retrieval_en_results.json` | B1 baseline, old Snellius run (stale dataset path) |
| `table_retrieval_b1_en_v2.json` | B1 baseline re-run on current merged test set |
| `table_retrieval_kg_units_en_results_v3.json` | B2 KG-enriched collection (kg_units), Snellius run — source of the 0.4658/0.8860 numbers in the meeting recap |
| `table_retrieval_kg_units_b2_en_results.json` | B2 alt collection variant (kg_units_b2), Snellius run |
| `table_retrieval_llm_reranker_kg_units_en_results.json` | LLM reranker on top of B2 kg_units |
| `table_retrieval_nl_results.json` | B1 baseline on Dutch (NL) test set |
| `alpha_tests/` | KG reranker alpha sweep (α = 0.5 → 1000); best result at α=25 (+8.8pp acc@1 over B1) |
