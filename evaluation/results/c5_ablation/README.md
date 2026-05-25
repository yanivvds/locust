# C5 Ablation — Version History (v4–v9)

Iterative development runs for the execution-based generator (C5), dev set unless marked test.
All runs use Gemma 4 31B or GPT-4.5-mini on the EN dev set (307 questions).

The `execution_loop_*` files are the earliest C5 prototypes (v1–v3, before the versioned naming convention).

Best version in this group: **v8** — tightened correction loop, aggregation guidance.
v8 test results: obsF1=0.592, msrF1=0.842, dimF1=0.643, EX_L=0.131.

See `documentation/c5_version_history.md` for the full version changelog.
