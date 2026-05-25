"""
Recall failure analysis for ColBERT table retrieval.

Runs the ColBERT retriever on the full English test set and identifies which
questions fail to have the correct table in the top-k candidate pool. For each
failure, collects diagnostic features to guide which recall-improvement strategy
(B5 RRF, B6 multi-query, B7 SKOS escape) is most likely to help.

Usage (Snellius):
    env PYTHONPATH=. LANGUAGE=en python3 analysis/recall_failure_analysis.py \\
        --checkpoint StatisticsNetherlands/GECKOv2-ColBERT-EN \\
        --collection_variant kg_units \\
        --k 10 \\
        --output documentation/history/b5_recall_failure_analysis.md

Run after the ColBERT index is built. Saves a Markdown report with per-failure
diagnostics and a summary breakdown by failure type.
"""
import argparse
import json
import re
import sys
import os
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from models.retrievers.colbert.colbert_retriever import ColBERTRetriever
from odata_graph import engine
from utils.global_functions import load_dataset, parse_for_table_id


FAILURE_LABELS = ("vocab_gap", "near_duplicate", "other")


def tokenize(text: str) -> set:
    return set(re.findall(r'\b[a-z]{3,}\b', text.lower()))


def token_overlap(a: str, b: str) -> float:
    ta, tb = tokenize(a), tokenize(b)
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


def get_table_title(table_id: str, title_cache: dict) -> str:
    return title_cache.get(table_id, {}).get("title", table_id)


def classify_failure(question: str, gold_id: str, top1_id: str, title_cache: dict) -> str:
    """Heuristic failure labelling.

    - near_duplicate: top-1 is a different table with high title overlap to gold
    - vocab_gap: gold table title has low token overlap with the question
    - other: everything else
    """
    gold_title  = get_table_title(gold_id, title_cache)
    top1_title  = get_table_title(top1_id, title_cache) if top1_id else ""

    q_gold_overlap  = token_overlap(question, gold_title)
    gold_top1_overlap = token_overlap(gold_title, top1_title)

    if gold_top1_overlap > 0.6:
        return "near_duplicate"
    if q_gold_overlap < 0.15:
        return "vocab_gap"
    return "other"


def run_analysis(checkpoint: str, collection_variant: str, k: int) -> list[dict]:
    retriever = ColBERTRetriever(
        checkpoint=checkpoint,
        mode='table',
        collection_variant=collection_variant,
    )

    dataset = load_dataset(config.TEST_QA_FILE)

    # Bulk-fetch all table titles for diagnosis
    title_query = """
        PREFIX dcat: <http://www.w3.org/ns/dcat#>
        PREFIX dct:  <http://purl.org/dc/terms/>
        SELECT DISTINCT ?id ?title WHERE {
            ?s a dcat:Dataset .
            ?s dct:identifier ?id .
            OPTIONAL { ?s dct:title ?title }
        }
    """
    rows = engine.select(title_query)
    title_cache = {r['id']['value']: {'title': r.get('title', {}).get('value', '')}
                   for r in rows}

    failures = []
    total = 0

    for item in dataset:
        question = item.question
        gold_ids = parse_for_table_id(item.sql, 'sql')
        if not gold_ids:
            continue
        gold_id = list(gold_ids)[0]
        total += 1

        predicted = list(retriever.retrieve_tables(question, k=k).keys())

        if gold_id not in predicted:
            top1 = predicted[0] if predicted else None
            label = classify_failure(question, gold_id, top1, title_cache)

            # token overlap features
            gold_title  = get_table_title(gold_id, title_cache)
            top1_title  = get_table_title(top1, title_cache) if top1 else ""
            q_gold_ov   = round(token_overlap(question, gold_title), 3)
            gold_t1_ov  = round(token_overlap(gold_title, top1_title), 3)

            failures.append({
                "question":      question,
                "gold_id":       gold_id,
                "gold_title":    gold_title,
                "top1_id":       top1,
                "top1_title":    top1_title,
                "top10_ids":     predicted[:k],
                "top10_titles":  [get_table_title(t, title_cache) for t in predicted[:k]],
                "q_gold_overlap":  q_gold_ov,
                "gold_top1_overlap": gold_t1_ov,
                "label":         label,
            })

    print(f"Total: {total}  Failures@{k}: {len(failures)}")
    return failures, total


def write_markdown(failures: list[dict], total: int, k: int, output_path: str):
    label_counts = Counter(f["label"] for f in failures)
    lines = [
        f"# Recall Failure Analysis — ColBERT acc@{k}",
        "",
        f"**Total questions:** {total}  ",
        f"**Failures at @{k}:** {len(failures)} ({100*len(failures)/total:.1f}%)",
        "",
        "## Breakdown by failure type",
        "",
        "| Type | Count | % of failures |",
        "|------|-------|--------------|",
    ]
    for label in FAILURE_LABELS:
        n = label_counts.get(label, 0)
        lines.append(f"| {label} | {n} | {100*n/max(len(failures),1):.0f}% |")

    lines += [
        "",
        "## Per-failure details",
        "",
    ]

    for i, f in enumerate(failures, 1):
        lines += [
            f"### {i}. [{f['label'].upper()}] {f['question']}",
            f"- **Gold table:** `{f['gold_id']}` — {f['gold_title']}",
            f"- **Top-1 retrieved:** `{f['top1_id']}` — {f['top1_title']}",
            f"- **Q↔gold title overlap:** {f['q_gold_overlap']}",
            f"- **Gold↔top-1 title overlap:** {f['gold_top1_overlap']}",
            f"- **Top-10 retrieved:**",
        ]
        for rank, (tid, title) in enumerate(zip(f['top10_ids'], f['top10_titles']), 1):
            lines.append(f"  {rank}. `{tid}` — {title}")
        lines.append("")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Saved report to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ColBERT recall failure analysis.")
    parser.add_argument("--checkpoint", required=True,
                        help="ColBERT checkpoint (HuggingFace ID or local path)")
    parser.add_argument("--collection_variant", default="kg_units",
                        help="Collection variant used when building the ColBERT index")
    parser.add_argument("--k", type=int, default=10,
                        help="Candidate pool size to evaluate recall at")
    parser.add_argument("--output", default="documentation/history/b5_recall_failure_analysis.md",
                        help="Path to write the Markdown report")
    args = parser.parse_args()

    failures, total = run_analysis(args.checkpoint, args.collection_variant, args.k)
    write_markdown(failures, total, args.k, args.output)
