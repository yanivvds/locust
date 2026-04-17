"""
Retrieval failure analysis for ColBERT kg_units variant.

Runs retrieval for all test questions, logs per-question hit/miss,
then breaks down failures by question type, theme, and most-confused tables.

Usage:
    CUDA_VISIBLE_DEVICES="" env PYTHONPATH=. LANGUAGE=en python3 \
        evaluation/analyze_retrieval_failures.py \
        --checkpoint StatisticsNetherlands/GECKOv2-ColBERT-EN \
        --collection_variant kg_units \
        --k 10
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict

import config  # noqa: sets up env vars
from utils.global_functions import load_dataset, load_model_from_path
from evaluation.evaluate_table_retrieval import parse_for_table_id


# ── sexp helpers ──────────────────────────────────────────────────────────────

def _sexp_type(sexp: str) -> str:
    """Infer question type from the outermost operator in the s-expression."""
    m = re.match(r'\((\w+)\s', sexp.strip())
    return m.group(1).upper() if m else 'UNKNOWN'


def _sexp_theme(sexp: str) -> str:
    """
    Heuristic: extract theme from table ID suffix.
    CBS table IDs end in letters that loosely map to themes (ENG=English, NED=Dutch, etc.)
    but the numeric prefix carries the real theme. Use a lookup if available, else 'UNKNOWN'.
    """
    ids = re.findall(r'\b([0-9]{5,}[A-Z]{2,4})\b', sexp, re.IGNORECASE)
    return ids[0].upper() if ids else 'UNKNOWN'


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyse retrieval failures.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--collection_variant", default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--query_type", default="sql", choices=["sql", "sexp"])
    parser.add_argument("--dataset_path", default=config.TEST_QA_FILE)
    parser.add_argument("--output", default="evaluation/results/failure_analysis.json")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_path)

    model_kwargs = {"checkpoint": args.checkpoint, "mode": "table"}
    if args.collection_variant:
        model_kwargs["collection_variant"] = args.collection_variant

    print(f"Loading retriever (checkpoint={args.checkpoint}, variant={args.collection_variant})...")
    retriever = load_model_from_path(
        "models/retrievers/colbert/colbert_retriever.py", **model_kwargs
    )

    hits, misses = [], []
    type_hits = defaultdict(list)
    type_misses = defaultdict(list)
    confused_tables = Counter()   # tables retrieved instead of correct one
    missed_tables = Counter()     # correct tables never retrieved

    for item in dataset:
        question = item.question
        query = item[args.query_type]
        gold = parse_for_table_id(query, args.query_type)
        if not gold:
            continue

        predicted = list(retriever.retrieve_tables(question, k=args.k).keys())
        qtype = _sexp_type(item.sexp) if hasattr(item, 'sexp') else 'UNKNOWN'

        hit = any(g in predicted for g in gold)
        record = {
            "question": question,
            "gold_table": gold,
            "predicted_top10": predicted,
            "hit": hit,
            "type": qtype,
        }

        if hit:
            hits.append(record)
            type_hits[qtype].append(record)
        else:
            misses.append(record)
            type_misses[qtype].append(record)
            for g in gold:
                missed_tables[g] += 1
            for p in predicted:
                if p not in gold:
                    confused_tables[p] += 1

    total = len(hits) + len(misses)
    acc = len(hits) / total if total else 0

    print(f"\n{'='*60}")
    print(f"  acc@{args.k}: {acc:.4f}  ({len(hits)}/{total})")
    print(f"  Missed: {len(misses)} questions")
    print(f"{'='*60}\n")

    # Breakdown by question type
    all_types = sorted(set(list(type_hits.keys()) + list(type_misses.keys())))
    print("Breakdown by question type:")
    print(f"  {'Type':<12} {'Hit':>5} {'Miss':>5} {'Acc':>7}")
    print(f"  {'-'*32}")
    for t in all_types:
        h = len(type_hits[t])
        m = len(type_misses[t])
        a = h / (h + m) if (h + m) else 0
        print(f"  {t:<12} {h:>5} {m:>5} {a:>7.1%}")

    # Most-missed correct tables
    print(f"\nTop 15 most-missed correct tables (correct table never in top {args.k}):")
    for table, count in missed_tables.most_common(15):
        print(f"  {table:<20} missed {count}x")

    # Most-confused tables (retrieved instead of correct)
    print(f"\nTop 15 most-confused tables (retrieved when wrong):")
    for table, count in confused_tables.most_common(15):
        print(f"  {table:<20} wrongly retrieved {count}x")

    # Sample missed questions
    print(f"\nSample of 10 missed questions:")
    for r in misses[:10]:
        print(f"  [{r['type']}] {r['question'][:80]}")
        print(f"         gold={r['gold_table']}  top3={r['predicted_top10'][:3]}")
        print()

    # Save full results
    output = {
        "variant": args.collection_variant or "baseline",
        "k": args.k,
        "accuracy": acc,
        "total": total,
        "n_hits": len(hits),
        "n_misses": len(misses),
        "by_type": {
            t: {
                "hits": len(type_hits[t]),
                "misses": len(type_misses[t]),
                "acc": len(type_hits[t]) / (len(type_hits[t]) + len(type_misses[t]))
            }
            for t in all_types
        },
        "most_missed_tables": missed_tables.most_common(30),
        "most_confused_tables": confused_tables.most_common(30),
        "missed_questions": misses,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Full results saved to {args.output}")


if __name__ == "__main__":
    main()
