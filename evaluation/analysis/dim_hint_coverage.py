"""
Script 3 — Hint coverage analysis.

For each dim error (extra or missing), classifies the failure as:
  - hint_gap:        correct code was never in probe_hits (retrieval/injection failure)
  - model_ignored:   correct code WAS in probe_hits but model didn't use it
  - hallucinated:    extra code wasn't in probe_hits either (pure hallucination)
  - hinted_wrong:    extra code WAS in probe_hits (model picked a wrong hinted code)

Requires: evaluation/analysis/output/dim_per_question.csv from Script 1.

Run:
    env PYTHONPATH=. LANGUAGE=en python3 evaluation/analysis/dim_hint_coverage.py

Output:
    evaluation/analysis/output/dim_hint_coverage.csv
    Console: 3-way failure breakdown
"""
import csv
import json
import os
import re
from collections import defaultdict

OUT_JSON  = "evaluation/results/c5_ablation/c5_v8_gemma_test_out.json"
INPUT_CSV = "evaluation/analysis/output/dim_per_question.csv"
OUTPUT_CSV = "evaluation/analysis/output/dim_hint_coverage.csv"


def parse_codes_from_field(field: str) -> list[tuple[str, str, str]]:
    """
    Parse 'CODE [dim_col] (label) | CODE2 [dim_col2] (label2)' into
    list of (code, dim_col, label). Handles bare codes too.
    """
    result = []
    for part in field.split(" | "):
        part = part.strip()
        if not part:
            continue
        m = re.match(r'^(\S+)\s+\[([^\]]+)\]\s+\(([^)]+)\)', part)
        if m:
            result.append((m.group(1), m.group(2), m.group(3)))
        else:
            # bare code, no annotation
            code = part.split()[0]
            result.append((code, "", ""))
    return result


def main():
    with open(OUT_JSON) as f:
        outputs = json.load(f)

    with open(INPUT_CSV, newline="") as f:
        question_rows = list(csv.DictReader(f))

    output_rows = []
    stats = defaultdict(int)

    for row in question_rows:
        question    = row["question"]
        out_rec     = outputs.get(question, {})
        probe_hits  = out_rec.get("probe_hits") or {}  # {dim_col: [code, ...]}

        # --- Missing codes (false negatives) ---
        for code, dim_col, label in parse_codes_from_field(row["missing_codes"]):
            if not code:
                continue
            probe_for_col = probe_hits.get(dim_col, [])
            if code in probe_for_col:
                failure_type = "model_ignored"
            else:
                failure_type = "hint_gap"

            stats[f"missing_{failure_type}"] += 1
            stats["missing_total"] += 1
            output_rows.append({
                "question":     question[:80],
                "sexp_type":    row["sexp_type"],
                "error_dir":    "missing",
                "code":         code,
                "dim_col":      dim_col,
                "label":        label,
                "in_probe_hits": code in probe_for_col,
                "failure_type": failure_type,
            })

        # --- Extra codes (false positives) ---
        for code, dim_col, label in parse_codes_from_field(row["extra_codes"]):
            if not code:
                continue
            probe_for_col = probe_hits.get(dim_col, [])
            if code in probe_for_col:
                failure_type = "hinted_wrong"
            else:
                failure_type = "hallucinated"

            stats[f"extra_{failure_type}"] += 1
            stats["extra_total"] += 1
            output_rows.append({
                "question":     question[:80],
                "sexp_type":    row["sexp_type"],
                "error_dir":    "extra",
                "code":         code,
                "dim_col":      dim_col,
                "label":        label,
                "in_probe_hits": code in probe_for_col,
                "failure_type": failure_type,
            })

    # Write CSV
    os.makedirs("evaluation/analysis/output", exist_ok=True)
    fieldnames = ["question", "sexp_type", "error_dir", "code", "dim_col",
                  "label", "in_probe_hits", "failure_type"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Saved {len(output_rows)} error rows → {OUTPUT_CSV}\n")

    # --- Console summary ---
    mt = stats["missing_total"]
    et = stats["extra_total"]

    print("=== MISSING dimension codes (false negatives) ===")
    print(f"  Total missing codes:  {mt}")
    if mt:
        mi = stats["missing_hint_gap"]
        mm = stats["missing_model_ignored"]
        print(f"  hint_gap      (code never in probe_hits): {mi:>4}  ({100*mi/mt:.1f}%)")
        print(f"  model_ignored (code was in probe_hits):   {mm:>4}  ({100*mm/mt:.1f}%)")

    print("\n=== EXTRA dimension codes (false positives) ===")
    print(f"  Total extra codes:    {et}")
    if et:
        eh = stats["extra_hallucinated"]
        ew = stats["extra_hinted_wrong"]
        print(f"  hallucinated  (not in probe_hits at all):  {eh:>4}  ({100*eh/et:.1f}%)")
        print(f"  hinted_wrong  (was in probe_hits, picked wrong one): {ew:>4}  ({100*ew/et:.1f}%)")

    # --- By question type ---
    print("\n=== Failure breakdown by question type ===")
    by_type: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    for r in output_rows:
        by_type[r["sexp_type"]][r["failure_type"]] += 1
        by_type[r["sexp_type"]]["total"] += 1

    header = f"{'Type':<12} {'Total':>6} {'hint_gap':>10} {'model_ignored':>15} {'hallucinated':>14} {'hinted_wrong':>14}"
    print(header)
    print("-" * len(header))
    for stype in sorted(by_type):
        t = by_type[stype]
        print(
            f"{stype:<12} {t['total']:>6}"
            f"  {t['hint_gap']:>6} ({100*t['hint_gap']/t['total'] if t['total'] else 0:>4.0f}%)"
            f"  {t['model_ignored']:>6} ({100*t['model_ignored']/t['total'] if t['total'] else 0:>4.0f}%)"
            f"  {t['hallucinated']:>6} ({100*t['hallucinated']/t['total'] if t['total'] else 0:>4.0f}%)"
            f"  {t['hinted_wrong']:>6} ({100*t['hinted_wrong']/t['total'] if t['total'] else 0:>4.0f}%)"
        )

    # --- Dim columns most affected ---
    from collections import Counter
    col_counts = Counter(r["dim_col"] for r in output_rows if r["error_dir"] == "extra" and r["dim_col"])
    print("\n=== Top 10 dimension columns with most extra codes ===")
    print(f"{'Dim col':<35} {'Extra codes':>12}")
    print("-" * 50)
    for col, count in col_counts.most_common(10):
        print(f"{col:<35} {count:>12}")


if __name__ == "__main__":
    main()
