# C5 v11 Sprint (v10–v11)

Results from the v11 sprint exploring schema-linking precision gates. All negative results —
no variant improved over v11base (= v8) on the full test set.

| Variant | Technique | Outcome |
|---------|-----------|---------|
| v10 | Lexical gate for dimension codes | Failed — dimF1 unchanged |
| v11a | DIVER existence gate (verify SKOS codes in parquet) | Failed — dimF1 unchanged |
| v11b | RSL-SQL binary selection (hint-free retry) | Failed — dimF1 unchanged |
| v11c | CHASE-SQL N-best sampling + execution-guided selection | Marginal gain dev, no gain test |
| v11d | Additional variant | Failed |
| v11base | v8 as default (recommended config) | Current best |

Qwen36 files (`_qwen36_`) test Qwen 3.6 35B MoE as an alternative model.
The persistent dimF1 ≈ 0.643 across all variants pins the root cause as parametric
(model injects CBS codes from pretraining memory, not from the prompt).
