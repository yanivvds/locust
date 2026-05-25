"""
Deep comparative analysis of LOCuST generator versions v8–v15 vs enterprise models.

Sections
--------
  1. Version comparison table (v8-v15 + enterprise)
  2. Failure mode taxonomy (from existing analysis CSVs)
  3. Enterprise SQL structural analysis (WHERE vs PIVOT filter placement)
  4. Intervention simulation on v8 outputs (no model re-run)
  5. Ranked recommendations with projected dimF1

Run from locust/ directory:
    env PYTHONPATH=. LANGUAGE=en python3 evaluation/analysis/deep_comparative_analysis.py
"""

import ast
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import sqlglot
from rapidfuzz import fuzz

BASE = Path(__file__).parent.parent.parent  # locust/
OUTPUT_DIR = BASE / "evaluation" / "analysis" / "output"
RESULTS_DIR = BASE / "evaluation" / "results"
ENTERPRISE_DIR = RESULTS_DIR / "enterprise_models"
DATA_DIR = BASE / "data" / "qa_pairs" / "en"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _f1(n_correct: int, n_pred: int, n_gt: int) -> float:
    if n_pred == 0 and n_gt == 0:
        return 1.0
    if n_pred == 0 or n_gt == 0:
        return 0.0
    p = n_correct / n_pred
    r = n_correct / n_gt
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _load_res(path: str | Path) -> dict:
    """Load a *_res.json result file and return lenient selection metrics + error counts."""
    with open(path) as f:
        d = json.load(f)
    sel = d.get("metrics", {}).get("selection_metrics", {}).get("lenient", {})
    ea = d.get("error_analysis", {})
    return {
        "msrF1": sel.get("measure_f1", 0.0),
        "dimF1": sel.get("dimension_f1", 0.0),
        "obsF1": sel.get("observation_f1", 0.0),
        "extra_d": ea.get("extra_dimensions", 0),
        "miss_d": ea.get("missing_dimensions", 0),
        "n": d.get("total_questions", 0),
    }


def _load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_out(path: str | Path) -> dict:
    """Load a *_out.json output file mapping question -> {query, ...}."""
    with open(path) as f:
        return json.load(f)


def _sep(char="─", width=80):
    print(char * width)


def _header(title: str):
    _sep()
    print(f"  {title}")
    _sep()


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Version comparison table
# ─────────────────────────────────────────────────────────────────────────────

def section1_version_comparison():
    _header("SECTION 1 — Version Comparison Table")

    versions = [
        ("v8 (original)",    RESULTS_DIR / "c5_ablation" / "c5_v8_gemma_test_res.json"),
        ("v12 baseline",     RESULTS_DIR / "c5_v12" / "c5_v12_baseline_gemma_test_res.json"),
        ("v12 pruning",      RESULTS_DIR / "c5_v12" / "c5_v12_pruning_gemma_test_res.json"),
        ("v13",              RESULTS_DIR / "c5_v13" / "c5_v13_gemma_test_res.json"),
        ("v14+pruning",      RESULTS_DIR / "c5_v14" / "c5_v14_pruning_gemma_test_res.json"),
        ("v15 (v8+prune)",   RESULTS_DIR / "c5_v15" / "c5_v15_pruning_gemma_test_res.json"),
        ("v8 repro",         RESULTS_DIR / "c5_v15" / "c5_v8_repro_gemma_test_res.json"),
    ]
    enterprise = [
        ("Claude 4.5 (R)",   ENTERPRISE_DIR / "claude_query-only_sql_en_results.json"),
        ("Gemini 2.5 (R)",   ENTERPRISE_DIR / "gemini_query-only_sql_en_results.json"),
        ("Claude 4.5+reason (R)", ENTERPRISE_DIR / "claude_query-only_sql_en_reasoning_results.json"),
    ]

    header = f"{'Version':<28} {'msrF1':>7} {'dimF1':>7} {'obsF1':>7} {'extra_d':>8} {'miss_d':>7} {'n':>5}"
    print(header)
    print("-" * len(header))

    for name, path in versions:
        if not Path(path).exists():
            print(f"  {name:<26} [FILE NOT FOUND: {path}]")
            continue
        m = _load_res(path)
        print(f"  {name:<26} {m['msrF1']:>7.4f} {m['dimF1']:>7.4f} {m['obsF1']:>7.4f} {m['extra_d']:>8} {m['miss_d']:>7} {m['n']:>5}")

    print(f"\n  {'--- Enterprise (with retrieval) ---':<30}")
    for name, path in enterprise:
        if not Path(path).exists():
            print(f"  {name:<26} [NO RES.JSON — from thesis image]")
            continue
        m = _load_res(path)
        print(f"  {name:<26} {m['msrF1']:>7.4f} {m['dimF1']:>7.4f} {m['obsF1']:>7.4f} {m['extra_d']:>8} {m['miss_d']:>7} {m['n']:>5}")

    print("\n  (GPT-5.1 with retrieval: msrF1≈0.81, dimF1≈0.84 — from thesis image, no res.json)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Failure mode taxonomy
# ─────────────────────────────────────────────────────────────────────────────

def section2_failure_taxonomy():
    _header("SECTION 2 — Failure Mode Taxonomy")

    coverage_path = OUTPUT_DIR / "dim_hint_coverage.csv"
    memory_path = OUTPUT_DIR / "dim_parametric_memory.csv"

    if not coverage_path.exists() or not memory_path.exists():
        print("  [SKIP] dim_hint_coverage.csv or dim_parametric_memory.csv not found.")
        print("  Run evaluation/analysis/dim_hint_coverage.py and dim_parametric_memory.py first.")
        return

    cov_df = pd.read_csv(coverage_path)
    mem_df = pd.read_csv(memory_path)

    # ── 2a. Hint coverage breakdown ──
    print("\n  2a. Failure type distribution (dim_hint_coverage.csv)")
    print(f"  Total error instances: {len(cov_df)}")
    ft_counts = cov_df["failure_type"].value_counts()
    for ft, n in ft_counts.items():
        pct = 100 * n / len(cov_df)
        print(f"    {ft:<20} {n:>6}  ({pct:.1f}%)")

    # Cross-tab: failure_type × sexp_type
    print("\n  2a-ii. failure_type × sexp_type cross-tab (counts):")
    ct = pd.crosstab(cov_df["failure_type"], cov_df.get("sexp_type", cov_df.iloc[:, 1]))
    print(ct.to_string(max_cols=20))

    # ── 2b. Parametric memory breakdown ──
    print("\n  2b. Parametric memory breakdown (dim_parametric_memory.csv)")
    print(f"  Total extra code instances: {len(mem_df)}")
    cat_counts = mem_df["category"].value_counts()
    for cat, n in cat_counts.items():
        pct = 100 * n / len(mem_df)
        exists = str(n)
        print(f"    {cat:<20} {n:>6}  ({pct:.1f}%)")

    # Map exists_in_parquet to bool robustly
    mem_df["exists_bool"] = mem_df["exists_in_parquet"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    n_invented = (mem_df["exists_bool"] == False).sum()
    n_real_wrong = (mem_df["exists_bool"] == True).sum()
    total = len(mem_df)
    print(f"\n  => {n_invented}/{total} ({100*n_invented/total:.1f}%) extra codes are invented")
    print(f"     (don't exist in the referenced parquet at all)")
    print(f"  => {n_real_wrong}/{total} ({100*n_real_wrong/total:.1f}%) are real-but-wrong")
    print(f"     (valid CBS codes that exist in parquet, just irrelevant to this question)")

    # Top hallucinated dimension columns
    print("\n  2c. Top dim_col by number of extra codes:")
    col_counts = mem_df["dim_col"].value_counts().head(15)
    for col, n in col_counts.items():
        invented_here = ((mem_df["dim_col"] == col) & (mem_df["exists_bool"] == False)).sum()
        print(f"    {str(col):<35} {n:>5} total  ({invented_here} invented)")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Enterprise SQL structural analysis
# ─────────────────────────────────────────────────────────────────────────────

def _count_sql_patterns(sql: str) -> dict:
    """Count WHERE IN and PIVOT FOR IN clauses in a SQL string using regex."""
    if not sql:
        return {"where_in": 0, "pivot_for_in": 0, "total_in": 0, "has_unpivot": 0}

    # Strip Measure and Value from counts (those are always expected)
    sql_upper = sql.upper()

    # PIVOT FOR ... IN (...) patterns
    pivot_for_pattern = re.compile(r'PIVOT\s*\(.*?FOR\s+\w+\s+IN\s*\(', re.IGNORECASE | re.DOTALL)
    # Simple IN clause (WHERE or inline) excluding MEASURE/VALUE
    where_in_pattern = re.compile(
        r'\b(?!MEASURE\b)(?!VALUE\b)(\w+)\s+IN\s*\(', re.IGNORECASE
    )

    pivot_matches = len(pivot_for_pattern.findall(sql))
    all_in = where_in_pattern.findall(sql)
    # Filter out Measure and Value
    non_measure_in = [col for col in all_in if col.upper() not in ('MEASURE', 'VALUE', 'FOR')]

    has_unpivot = 1 if "UNPIVOT" in sql_upper else 0

    return {
        "where_in": len(non_measure_in) - pivot_matches,
        "pivot_for_in": pivot_matches,
        "total_in": len(non_measure_in),
        "has_unpivot": has_unpivot,
    }


def section3_enterprise_structural():
    _header("SECTION 3 — Enterprise SQL Structural Analysis")

    v8_out_path = RESULTS_DIR / "c5_ablation" / "c5_v8_gemma_test_out.json"
    enterprise_paths = {
        "Claude 4.5": ENTERPRISE_DIR / "claude_query-only_sql_en.json",
        "Gemini 2.5": ENTERPRISE_DIR / "gemini_query-only_sql_en.json",
        "GPT-5.1":    ENTERPRISE_DIR / "gpt5_query-only_sql_en.json",
    }

    if not v8_out_path.exists():
        print("  [SKIP] v8 out.json not found.")
        return

    v8_out = _load_out(v8_out_path)
    v8_questions = set(v8_out.keys())

    print("\n  Comparing SQL structure: Gemma v8 vs enterprise models")
    print(f"  (matching questions only — same question must appear in both outputs)\n")

    header = f"  {'Model':<22} {'matched_q':>10} {'avg_extra_d':>12} {'avg_where_in':>13} {'avg_pivot_in':>13} {'unpivot_%':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    # Gemma stats on all questions
    gemma_stats = []
    for q, entry in v8_out.items():
        sql = entry.get("query", "")
        patterns = _count_sql_patterns(sql)
        gemma_stats.append(patterns)

    if gemma_stats:
        avg_where = sum(s["where_in"] for s in gemma_stats) / len(gemma_stats)
        avg_pivot = sum(s["pivot_for_in"] for s in gemma_stats) / len(gemma_stats)
        avg_unpivot_pct = 100 * sum(s["has_unpivot"] for s in gemma_stats) / len(gemma_stats)
        print(f"  {'Gemma v8 (all)':<22} {len(gemma_stats):>10} {'—':>12} {avg_where:>13.2f} {avg_pivot:>13.2f} {avg_unpivot_pct:>9.1f}%")

    for model_name, ent_path in enterprise_paths.items():
        if not ent_path.exists():
            print(f"  {model_name:<22} [FILE NOT FOUND]")
            continue
        ent_out = _load_out(ent_path)

        # Match on question text
        matched_q = [q for q in ent_out if q in v8_questions]
        if not matched_q:
            print(f"  {model_name:<22} [NO MATCHING QUESTIONS]")
            continue

        ent_stats = []
        for q in matched_q:
            sql = ent_out[q].get("query", "") if isinstance(ent_out[q], dict) else str(ent_out[q])
            ent_stats.append(_count_sql_patterns(sql))

        avg_where = sum(s["where_in"] for s in ent_stats) / len(ent_stats)
        avg_pivot = sum(s["pivot_for_in"] for s in ent_stats) / len(ent_stats)
        avg_unpivot_pct = 100 * sum(s["has_unpivot"] for s in ent_stats) / len(ent_stats)
        print(f"  {model_name:<22} {len(matched_q):>10} {'—':>12} {avg_where:>13.2f} {avg_pivot:>13.2f} {avg_unpivot_pct:>9.1f}%")

    # Matched comparison: Gemma vs Claude on same questions
    claude_path = ENTERPRISE_DIR / "claude_query-only_sql_en.json"
    if claude_path.exists():
        claude_out = _load_out(claude_path)
        matched = [q for q in claude_out if q in v8_questions]
        print(f"\n  Matched question analysis (Gemma v8 vs Claude 4.5, n={len(matched)}):")

        gemma_m, claude_m = [], []
        for q in matched:
            gemma_m.append(_count_sql_patterns(v8_out[q].get("query", "")))
            sql_c = claude_out[q].get("query", "") if isinstance(claude_out[q], dict) else str(claude_out[q])
            claude_m.append(_count_sql_patterns(sql_c))

        avg_g_where = sum(s["where_in"] for s in gemma_m) / max(1, len(gemma_m))
        avg_c_where = sum(s["where_in"] for s in claude_m) / max(1, len(claude_m))
        avg_g_pivot = sum(s["pivot_for_in"] for s in gemma_m) / max(1, len(gemma_m))
        avg_c_pivot = sum(s["pivot_for_in"] for s in claude_m) / max(1, len(claude_m))

        print(f"    Avg WHERE IN clauses:     Gemma={avg_g_where:.2f}  Claude={avg_c_where:.2f}")
        print(f"    Avg PIVOT FOR IN clauses: Gemma={avg_g_pivot:.2f}  Claude={avg_c_pivot:.2f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Intervention simulation
# ─────────────────────────────────────────────────────────────────────────────

def section4_intervention_simulation():
    _header("SECTION 4 — Intervention Simulation (on existing v8 outputs)")

    pq_path = OUTPUT_DIR / "dim_per_question.csv"
    mem_path = OUTPUT_DIR / "dim_parametric_memory.csv"
    jsonl_path = DATA_DIR / "complex_en_test.jsonl"

    if not pq_path.exists() or not mem_path.exists():
        print("  [SKIP] Required CSVs not found. Run dim_error_breakdown.py first.")
        return

    pq_df = pd.read_csv(pq_path)
    mem_df = pd.read_csv(mem_path)

    # Normalize exists_in_parquet to boolean
    mem_df["exists_bool"] = mem_df["exists_in_parquet"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )

    # Build per-question invented / real_wrong counts and labels
    invented_per_q: dict[str, int] = defaultdict(int)
    real_wrong_per_q: dict[str, list] = defaultdict(list)
    for _, row in mem_df.iterrows():
        q = str(row["question"])
        if row["exists_bool"] == False:
            invented_per_q[q] += 1
        else:
            real_wrong_per_q[q].append((str(row.get("extra_code", "")), str(row.get("label", ""))))

    # Load question text for semantic scoring (Intervention B)
    question_text: dict[str, str] = {}
    if jsonl_path.exists():
        for item in _load_jsonl(jsonl_path):
            question_text[item["question"]] = item["question"]
    else:
        print("  [WARN] complex_en_test.jsonl not found; Intervention B will be skipped.")

    # Baselines
    baseline_f1_list = []
    intervention_A_f1_list = []
    intervention_B_results: dict[int, list] = {40: [], 50: [], 60: [], 70: []}
    intervention_C_results: dict[int, list] = {40: [], 50: [], 60: [], 70: []}

    extra_baseline = 0
    extra_after_A = 0
    extra_after_C60 = 0

    for _, row in pq_df.iterrows():
        q = str(row["question"])
        n_gt = int(row.get("n_gt_dims", 0))
        n_pred = int(row.get("n_pred_dims", 0))
        n_extra = int(row.get("n_extra", 0))
        n_missing = int(row.get("n_missing", 0))
        n_correct = n_gt - n_missing  # codes correctly predicted

        baseline_f1 = _f1(n_correct, n_pred, n_gt)
        baseline_f1_list.append(baseline_f1)
        extra_baseline += n_extra

        # ── Intervention A: prune invented codes ──
        n_invented = invented_per_q.get(q, 0)
        # Clamp: can't prune more than n_extra
        n_pruned_A = min(n_invented, n_extra)
        new_pred_A = n_pred - n_pruned_A
        f1_A = _f1(n_correct, max(0, new_pred_A), n_gt)
        intervention_A_f1_list.append(f1_A)
        extra_after_A += (n_extra - n_pruned_A)

        # ── Intervention B: semantic pruning of real_wrong codes ──
        real_wrong = real_wrong_per_q.get(q, [])
        for threshold in intervention_B_results:
            n_pruned_B = 0
            for code, label in real_wrong:
                if not label or label == "nan":
                    # If no label, prune conservatively at high thresholds only
                    if threshold <= 50:
                        n_pruned_B += 1
                    continue
                score = fuzz.partial_ratio(q, label)
                if score < threshold:
                    n_pruned_B += 1
            n_pruned_B = min(n_pruned_B, n_extra)
            new_pred_B = n_pred - n_pruned_B
            f1_B = _f1(n_correct, max(0, new_pred_B), n_gt)
            intervention_B_results[threshold].append(f1_B)

        # ── Intervention C: A + B combined ──
        for threshold in intervention_C_results:
            n_pruned_B = 0
            for code, label in real_wrong:
                if not label or label == "nan":
                    if threshold <= 50:
                        n_pruned_B += 1
                    continue
                score = fuzz.partial_ratio(q, label)
                if score < threshold:
                    n_pruned_B += 1
            n_pruned_combined = min(n_pruned_A + n_pruned_B, n_extra)
            new_pred_C = n_pred - n_pruned_combined
            f1_C = _f1(n_correct, max(0, new_pred_C), n_gt)
            intervention_C_results[threshold].append(f1_C)

        extra_after_C60 += max(0, n_extra - min(n_pruned_A + sum(
            1 for code, label in real_wrong
            if (not label or label == "nan") or fuzz.partial_ratio(q, label) < 60
        ), n_extra))

    n_q = len(pq_df)
    baseline_dimf1 = sum(baseline_f1_list) / n_q
    A_dimf1 = sum(intervention_A_f1_list) / n_q

    print(f"\n  Simulation over {n_q} questions from dim_per_question.csv\n")
    print(f"  {'Intervention':<40} {'proj_dimF1':>11} {'Δ vs v8':>8} {'extra_after':>12} {'miss_after':>11}")
    print("  " + "-" * 85)

    v8_missing = int(pq_df["n_missing"].sum())

    print(f"  {'Baseline (v8)':<40} {baseline_dimf1:>11.4f} {'—':>8} {extra_baseline:>12} {v8_missing:>11}")
    print(f"  {'A: DuckDB existence prune':<40} {A_dimf1:>11.4f} {A_dimf1-baseline_dimf1:>+8.4f} {extra_after_A:>12} {v8_missing:>11}")

    best_B_threshold = max(intervention_B_results, key=lambda t: sum(intervention_B_results[t]) / n_q)
    for threshold, f1_list in sorted(intervention_B_results.items()):
        B_dimf1 = sum(f1_list) / n_q
        print(f"  {'B: Semantic prune (thresh='+str(threshold)+')':<40} {B_dimf1:>11.4f} {B_dimf1-baseline_dimf1:>+8.4f} {'—':>12} {v8_missing:>11}")

    for threshold, f1_list in sorted(intervention_C_results.items()):
        C_dimf1 = sum(f1_list) / n_q
        label = f"C: A+B combined (thresh={threshold})"
        print(f"  {label:<40} {C_dimf1:>11.4f} {C_dimf1-baseline_dimf1:>+8.4f} {'—':>12} {v8_missing:>11}")

    print(f"\n  Enterprise ceiling:")
    print(f"  {'Claude 4.5 (R)':<40} {'0.8449':>11} {'+0.2019':>8} {'115':>12} {'84':>11}")
    print(f"  {'Gemini 2.5 (R)':<40} {'0.8512':>11} {'+0.2082':>8} {'110':>12} {'90':>11}")

    # Per-type breakdown of intervention A
    print(f"\n  Intervention A — per question type dimF1 improvement:")
    print(f"  {'sexp_type':<12} {'n':>5} {'baseline':>10} {'after_A':>10} {'Δ':>8}")
    print("  " + "-" * 50)

    type_baseline: dict[str, list] = defaultdict(list)
    type_A: dict[str, list] = defaultdict(list)
    for i, (_, row) in enumerate(pq_df.iterrows()):
        t = str(row.get("sexp_type", "UNKNOWN"))
        type_baseline[t].append(baseline_f1_list[i])
        type_A[t].append(intervention_A_f1_list[i])

    for stype in sorted(type_baseline.keys()):
        nb = len(type_baseline[stype])
        mb = sum(type_baseline[stype]) / nb
        ma = sum(type_A[stype]) / nb
        print(f"  {stype:<12} {nb:>5} {mb:>10.4f} {ma:>10.4f} {ma-mb:>+8.4f}")

    # Save simulation CSV
    sim_rows = []
    for i, (_, row) in enumerate(pq_df.iterrows()):
        sim_rows.append({
            "question": row["question"],
            "sexp_type": row.get("sexp_type", ""),
            "n_gt_dims": row.get("n_gt_dims", 0),
            "n_pred_dims": row.get("n_pred_dims", 0),
            "n_extra": row.get("n_extra", 0),
            "n_missing": row.get("n_missing", 0),
            "baseline_dimf1": baseline_f1_list[i],
            "n_invented": invented_per_q.get(str(row["question"]), 0),
            "n_real_wrong": len(real_wrong_per_q.get(str(row["question"]), [])),
            "dimf1_after_A": intervention_A_f1_list[i],
            "dimf1_after_B60": intervention_B_results[60][i],
            "dimf1_after_C60": intervention_C_results[60][i],
        })
    sim_df = pd.DataFrame(sim_rows)
    out_path = OUTPUT_DIR / "intervention_simulation.csv"
    sim_df.to_csv(out_path, index=False)
    print(f"\n  Saved simulation results to {out_path}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Ranked recommendations
# ─────────────────────────────────────────────────────────────────────────────

def section5_recommendations():
    _header("SECTION 5 — Ranked Recommendations")

    sim_path = OUTPUT_DIR / "intervention_simulation.csv"
    if not sim_path.exists():
        print("  [SKIP] intervention_simulation.csv not found. Run Section 4 first.")
        return

    sim_df = pd.read_csv(sim_path)
    n = len(sim_df)

    baseline = sim_df["baseline_dimf1"].mean()
    A = sim_df["dimf1_after_A"].mean()
    B60 = sim_df["dimf1_after_B60"].mean()
    C60 = sim_df["dimf1_after_C60"].mean()

    print(f"""
  RECOMMENDED INTERVENTIONS (ranked by projected dimF1 improvement)
  ─────────────────────────────────────────────────────────────────

  #1 — DuckDB On-Demand Existence Verification (Intervention A)
       Projected dimF1: {A:.4f}  (Δ {A-baseline:+.4f} vs v8 baseline {baseline:.4f})

       Method: After generation, for each dimension IN-clause in the SQL, run:
           SELECT DISTINCT dim_col FROM parquet WHERE dim_col IN (codes) LIMIT 1
       Prune any code that returns 0 rows (doesn't exist in the target parquet).
       Skip Periods and other time dimensions (use _is_time_dim check).

       Why it works: {100*(sim_df['n_invented'].sum() / max(1, sim_df['n_extra'].sum())):.0f}% of extra codes are invented
       (don't exist in any row of the referenced parquet). DuckDB verification
       catches ALL of these regardless of whether the column was pre-probed.

       Implementation: Add _verify_dim_codes_duckdb(sql) → str to
       execution_loop_generator.py and wire at all 3 return paths.
       CLI flag: --use_duckdb_verify true
       Overhead: ~0.5-1s per question (DuckDB local query, much less than LLM call).

  #2 — Semantic Relevance Pruning of Real-But-Wrong Codes (Intervention B, thresh=60)
       Projected dimF1: {B60:.4f}  (Δ {B60-baseline:+.4f})

       Method: For codes that DO exist in the parquet but are still wrong for the
       question (real_but_wrong category), compute rapidfuzz.partial_ratio(question, code_label).
       Prune if score < 60.

       Risk: If the code label phrasing doesn't match the question wording (e.g.,
       "motor vehicle repair services" vs "garages"), will incorrectly prune.
       Implement with a conservative threshold (60) and only after confirming A.

  #3 — Combined A + B (Intervention C, thresh=60)
       Projected dimF1: {C60:.4f}  (Δ {C60-baseline:+.4f})

       Combined effect of both interventions.

  ENTERPRISE CEILING:
       Claude 4.5:  dimF1=0.8449  Δ+{0.8449-baseline:+.4f} vs Gemma v8
       Gemini 2.5:  dimF1=0.8512  Δ+{0.8512-baseline:+.4f} vs Gemma v8

  NOTE ON MISSING DIMS:
       All interventions only prune extras — recall is unchanged.
       Missing dims (currently {int(sim_df['n_missing'].sum())} total) require:
       - Better hint injection (higher probe_top_k_tables, lower col_label_threshold)
       - Improved code retrieval (SKOS fuzzy matching with lower threshold)
       These are separate experiments to improve recall independently.
""")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    section1_version_comparison()
    section2_failure_taxonomy()
    section3_enterprise_structural()
    section4_intervention_simulation()
    section5_recommendations()

    print("=" * 80)
    print("  Analysis complete.")
    print(f"  Outputs saved to: {OUTPUT_DIR}")
    print("=" * 80)
