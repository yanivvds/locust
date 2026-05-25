"""
Script 2 — Parametric memory hypothesis validation.

For each hallucinated (extra) dimension code, checks whether it is a valid CBS
code that actually exists in the relevant parquet file. If it does, Gemma recalled
a real code from training data but used it in the wrong context (parametric memory).
If it doesn't exist at all, it was fully invented.

Requires: evaluation/analysis/output/dim_per_question.csv from Script 1.

Run:
    env PYTHONPATH=. LANGUAGE=en python3 evaluation/analysis/dim_parametric_memory.py

Output:
    evaluation/analysis/output/dim_parametric_memory.csv
    Console: breakdown of real-but-wrong vs invented codes
"""
import ast
import csv
import json
import os
import re
from collections import defaultdict

import duckdb

OUT_JSON   = "evaluation/results/c5_ablation/c5_v8_gemma_test_out.json"
INPUT_CSV  = "evaluation/analysis/output/dim_per_question.csv"
OUTPUT_CSV = "evaluation/analysis/output/dim_parametric_memory.csv"
PARQUET_DIR = "data/en/odata3"


def table_id_from_sql(sql: str) -> str | None:
    m = re.search(r'(?<=[/\\])([^/\\]+)(?=\.parquet)', sql)
    return m.group(1) if m else None


def code_exists_in_parquet(table_id: str, dim_col: str, code: str) -> bool:
    """Check if a code exists in a specific column of a table's parquet file."""
    path = os.path.join(PARQUET_DIR, f"{table_id}.parquet")
    if not os.path.exists(path):
        return False
    try:
        result = duckdb.sql(
            f'SELECT 1 FROM read_parquet(\'{path}\') WHERE "{dim_col}" = \'{code}\' LIMIT 1'
        ).fetchall()
        return len(result) > 0
    except Exception:
        return False


def dim_col_from_code_str(code_str: str) -> tuple[str, str] | None:
    """Parse 'CODE [dim_col] (label)' format from Script 1 CSV."""
    m = re.match(r'^(\S+)\s+\[([^\]]+)\]', code_str.strip())
    if m:
        return m.group(1), m.group(2)
    # Fallback: just a bare code with no annotation
    code = code_str.strip().split()[0]
    return code, None


def main():
    with open(OUT_JSON) as f:
        outputs = json.load(f)

    # Read per-question CSV from Script 1
    with open(INPUT_CSV, newline="") as f:
        reader = csv.DictReader(f)
        question_rows = list(reader)

    output_rows = []
    stats = defaultdict(int)

    for row in question_rows:
        if not row["extra_codes"]:
            continue

        question = row["question"]
        out_rec  = outputs.get(question, {})
        pred_sql = out_rec.get("query", "")
        table_id = table_id_from_sql(pred_sql) if pred_sql else None

        for code_str in row["extra_codes"].split(" | "):
            code_str = code_str.strip()
            if not code_str:
                continue

            parsed = dim_col_from_code_str(code_str)
            if not parsed:
                continue
            code, dim_col = parsed

            # Extract label from the code_str if present
            label_m = re.search(r'\(([^)]+)\)$', code_str)
            label = label_m.group(1) if label_m else ""

            exists = False
            if table_id and dim_col:
                exists = code_exists_in_parquet(table_id, dim_col, code)

            category = "real_but_wrong" if exists else "invented"
            stats[category] += 1
            stats["total"] += 1

            output_rows.append({
                "question":          question[:80],
                "sexp_type":         row["sexp_type"],
                "extra_code":        code,
                "dim_col":           dim_col or "",
                "label":             label,
                "table_id":          table_id or "",
                "exists_in_parquet": exists,
                "category":          category,
            })

    # Write CSV
    os.makedirs("evaluation/analysis/output", exist_ok=True)
    fieldnames = ["question", "sexp_type", "extra_code", "dim_col", "label",
                  "table_id", "exists_in_parquet", "category"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Saved {len(output_rows)} extra-code rows → {OUTPUT_CSV}\n")

    # Console summary
    total = stats["total"]
    real  = stats["real_but_wrong"]
    inv   = stats["invented"]

    print("=== Parametric memory analysis ===")
    print(f"Total extra (hallucinated) codes analysed: {total}")
    if total:
        print(f"  real_but_wrong (valid CBS code, wrong context): {real:>4}  ({100*real/total:.1f}%)")
        print(f"  invented       (code does not exist in parquet): {inv:>4}  ({100*inv/total:.1f}%)")

    print("\n=== By question type ===")
    by_type = defaultdict(lambda: defaultdict(int))
    for r in output_rows:
        by_type[r["sexp_type"]][r["category"]] += 1
        by_type[r["sexp_type"]]["total"] += 1

    print(f"{'Type':<12} {'Total':>6} {'real_but_wrong':>16} {'invented':>10}")
    print("-" * 48)
    for stype in sorted(by_type):
        t = by_type[stype]
        print(f"{stype:<12} {t['total']:>6} {t['real_but_wrong']:>14}  ({100*t['real_but_wrong']/t['total'] if t['total'] else 0:.0f}%)  "
              f"{t['invented']:>6}  ({100*t['invented']/t['total'] if t['total'] else 0:.0f}%)")

    # Most commonly hallucinated codes
    from collections import Counter
    code_counts = Counter(r["extra_code"] for r in output_rows if r["category"] == "real_but_wrong")
    print("\n=== Top 10 most-hallucinated real codes ===")
    print(f"{'Code':<16} {'Dim col':<30} {'Label':<40} {'Count':>6}")
    print("-" * 96)
    for code, count in code_counts.most_common(10):
        sample = next((r for r in output_rows if r["extra_code"] == code), {})
        print(f"{code:<16} {sample.get('dim_col',''):<30} {sample.get('label',''):<40} {count:>6}")


if __name__ == "__main__":
    main()
