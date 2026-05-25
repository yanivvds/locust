"""
Script 1 — DimF1 per-question breakdown with KG labels.

For every question in the v8 test run, computes which dimension codes were
extra (hallucinated) and which were missing, then looks up their human-readable
labels from the knowledge graph so we can read examples directly.

Run:
    env PYTHONPATH=. LANGUAGE=en python3 evaluation/analysis/dim_error_breakdown.py

Output:
    evaluation/analysis/output/dim_per_question.csv  (one row per question)
    Console: summary table by error class and by s-expression type
"""
import csv
import json
import os
import re
from collections import defaultdict
from functools import lru_cache

from evaluation.metrics.selection_metrics import extract_sql_components, calculate_f1
from models.generators.code_resolver import SKOSCodeResolver
from odata_graph import engine
from s_expression import Table

OUT_JSON   = "evaluation/results/c5_ablation/c5_v8_gemma_test_out.json"
TEST_JSONL = "data/qa_pairs/en/complex_en_test.jsonl"
OUTPUT_DIR = "evaluation/analysis/output"
OUTPUT_CSV = f"{OUTPUT_DIR}/dim_per_question.csv"

resolver = SKOSCodeResolver(engine)


def sexp_type(sexp: str) -> str:
    m = re.match(r'\((\w+)', sexp.strip())
    return m.group(1) if m else 'UNKNOWN'


def table_id_from_sql(sql: str) -> str | None:
    m = re.search(r'(?<=[/\\])([^/\\]+)(?=\.parquet)', sql)
    return m.group(1) if m else None


@lru_cache(maxsize=512)
def code_label_map(table_id: str) -> dict:
    """Return {code: (dim_col, label)} for all non-geo/time dims of a table."""
    table = Table(table_id)
    try:
        dim_cols = resolver.get_schema_dim_labels(table)
    except Exception:
        return {}
    result = {}
    for dim_col in dim_cols:
        try:
            for code, label in engine.get_dimension_codes(table, dim_col).items():
                result[code] = (dim_col, label)
        except Exception:
            continue
    return result


def format_codes(codes: set, lookup: dict) -> str:
    parts = []
    for code in sorted(codes):
        if code in lookup:
            dim_col, label = lookup[code]
            parts.append(f"{code} [{dim_col}] ({label})")
        else:
            parts.append(code)
    return " | ".join(parts)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(OUT_JSON) as f:
        outputs = json.load(f)

    gold_records = []
    with open(TEST_JSONL) as f:
        for line in f:
            gold_records.append(json.loads(line.strip()))

    rows = []
    for rec in gold_records:
        question = rec["question"]
        gt_sql   = rec["sql"]
        stype    = sexp_type(rec.get("sexp", ""))

        out_rec  = outputs.get(question, {})
        pred_sql = out_rec.get("query", "")

        if not pred_sql:
            rows.append({
                "question": question, "sexp_type": stype,
                "n_gt_dims": 0, "n_pred_dims": 0,
                "n_extra": 0, "n_missing": 0,
                "extra_codes": "", "missing_codes": "",
                "dim_f1": 0.0, "error_class": "generation_fail",
            })
            continue

        _, gt_dims   = extract_sql_components(gt_sql)
        _, pred_dims = extract_sql_components(pred_sql)

        extra   = pred_dims - gt_dims
        missing = gt_dims  - pred_dims
        dim_f1  = calculate_f1(pred_dims, gt_dims)

        if extra and missing:
            eclass = "both"
        elif extra:
            eclass = "extra_only"
        elif missing:
            eclass = "missing_only"
        else:
            eclass = "perfect"

        # Build label lookup from both generated and gold table ids
        lookup = {}
        for sql in (pred_sql, gt_sql):
            tid = table_id_from_sql(sql)
            if tid:
                lookup.update(code_label_map(tid))

        rows.append({
            "question":    question,
            "sexp_type":   stype,
            "n_gt_dims":   len(gt_dims),
            "n_pred_dims": len(pred_dims),
            "n_extra":     len(extra),
            "n_missing":   len(missing),
            "extra_codes":   format_codes(extra,   lookup),
            "missing_codes": format_codes(missing, lookup),
            "dim_f1":      round(dim_f1, 4),
            "error_class": eclass,
        })

    # --- Write CSV ---
    fieldnames = ["question", "sexp_type", "n_gt_dims", "n_pred_dims",
                  "n_extra", "n_missing", "extra_codes", "missing_codes",
                  "dim_f1", "error_class"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows → {OUTPUT_CSV}\n")

    # --- Console: by error class ---
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["error_class"]].append(r["dim_f1"])

    print("=== By error class ===")
    print(f"{'Class':<20} {'Count':>6} {'Avg dimF1':>10}")
    print("-" * 40)
    for cls in ["perfect", "extra_only", "missing_only", "both", "generation_fail"]:
        vals = by_class.get(cls, [])
        avg  = sum(vals) / len(vals) if vals else 0
        print(f"{cls:<20} {len(vals):>6} {avg:>10.4f}")
    total = [r["dim_f1"] for r in rows]
    print(f"{'TOTAL':<20} {len(total):>6} {sum(total)/len(total):>10.4f}")

    # --- Console: by sexp type ---
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["sexp_type"]].append(r["dim_f1"])

    print("\n=== By question type ===")
    print(f"{'Type':<12} {'Count':>6} {'Avg dimF1':>10} {'% extra':>8} {'% missing':>10}")
    print("-" * 52)
    for stype in sorted(by_type):
        type_rows = [r for r in rows if r["sexp_type"] == stype]
        avg   = sum(r["dim_f1"] for r in type_rows) / len(type_rows)
        pextra   = 100 * sum(1 for r in type_rows if r["n_extra"]   > 0) / len(type_rows)
        pmissing = 100 * sum(1 for r in type_rows if r["n_missing"] > 0) / len(type_rows)
        print(f"{stype:<12} {len(type_rows):>6} {avg:>10.4f} {pextra:>7.1f}% {pmissing:>9.1f}%")

    # --- Top 10 worst questions ---
    print("\n=== 10 worst questions by dimF1 (non-fail) ===")
    worst = sorted([r for r in rows if r["error_class"] != "generation_fail"],
                   key=lambda r: r["dim_f1"])[:10]
    for r in worst:
        print(f"  [{r['dim_f1']:.3f}] [{r['sexp_type']}] {r['question'][:70]}")
        if r["extra_codes"]:
            print(f"    EXTRA:   {r['extra_codes'][:120]}")
        if r["missing_codes"]:
            print(f"    MISSING: {r['missing_codes'][:120]}")


if __name__ == "__main__":
    main()
