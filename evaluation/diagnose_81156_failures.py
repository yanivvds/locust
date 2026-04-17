"""
Targeted failure analysis for table 81156eng.

For each question where 81156eng is missed (or hit below rank 1), shows:
  - Which tables rank above it
  - Token overlap between query and each confounder's collection document body
  - Whether the correct table's schema labels contain the query terms (reranker signal)

Helps confirm whether table collision (near-identical descriptions) is the root cause,
and whether a KG schema-overlap reranker would help.

Usage:
    CUDA_VISIBLE_DEVICES="" env PYTHONPATH=. LANGUAGE=en python3 \
        evaluation/diagnose_81156_failures.py \
        --checkpoint StatisticsNetherlands/GECKOv2-ColBERT-EN \
        --collection_variant kg_units
"""
import argparse
import json
import re

import config
from utils.global_functions import load_dataset, load_model_from_path
from evaluation.evaluate_table_retrieval import parse_for_table_id
from models.retrievers.colbert.colbert_retriever import BASE_PATH

TARGET_TABLE = "81156eng"


def _tokens(text: str) -> set:
    return set(re.findall(r'\w+', text.lower()))


def main():
    parser = argparse.ArgumentParser(description=f"Diagnose retrieval failures for {TARGET_TABLE}.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--collection_variant", default="kg_units")
    parser.add_argument("--k", type=int, default=20,
                        help="Retrieve top-k (use >10 to see if target appears just outside top-10)")
    parser.add_argument("--query_type", default="sql", choices=["sql", "sexp"])
    parser.add_argument("--output", default=f"evaluation/results/diagnose_{TARGET_TABLE}.json")
    args = parser.parse_args()

    # Load collection bodies indexed by table_id
    checkpoint_slug = args.checkpoint.replace('/', '_')
    data_dir = f"{BASE_PATH}/{checkpoint_slug}"
    collection_path = f"{data_dir}/collection_table_{args.collection_variant}.tsv"
    map_path = f"{data_dir}/collection_table_{args.collection_variant}_node_pid_map.json"

    with open(collection_path) as f:
        coll_lines = [line.rstrip('\n') for line in f]
    with open(map_path) as f:
        node_pid_map = json.load(f)  # {table_id: pid}
    table_body = {tid: coll_lines[pid] for tid, pid in node_pid_map.items() if pid < len(coll_lines)}

    target_body_tokens = _tokens(table_body.get(TARGET_TABLE, ''))

    # Load node labels for schema-level overlap (reranker signal)
    labels_path = f"{data_dir}/{config.GRAPH_DB_REPO}_node_labels.json"
    schema_tokens = {}
    try:
        with open(labels_path) as f:
            node_labels = json.load(f)
        for key, data in node_labels.items():
            if data.get('type') == 'table':
                continue
            tid = data.get('table') or key.split('#')[0]
            text = data.get('prefLabel') or data.get('body', '')
            if tid not in schema_tokens:
                schema_tokens[tid] = set()
            schema_tokens[tid].update(_tokens(text))
        print(f"Loaded schema tokens for {len(schema_tokens)} tables from {labels_path}")
    except FileNotFoundError:
        print(f"WARNING: node_labels cache not found at {labels_path}; schema overlap analysis skipped")

    # Load retriever
    print(f"Loading retriever (checkpoint={args.checkpoint}, variant={args.collection_variant})...")
    retriever = load_model_from_path(
        "models/retrievers/colbert/colbert_retriever.py",
        checkpoint=args.checkpoint,
        mode="table",
        collection_variant=args.collection_variant,
    )

    dataset = load_dataset(config.TEST_QA_FILE)

    records = []
    for item in dataset:
        query = item[args.query_type]
        gold = parse_for_table_id(query, args.query_type)
        if TARGET_TABLE not in gold:
            continue

        question = item.question
        predicted = list(retriever.retrieve_tables(question, k=args.k).keys())
        hit_at_10 = TARGET_TABLE in predicted[:10]
        rank = (predicted.index(TARGET_TABLE) + 1) if TARGET_TABLE in predicted else None

        query_toks = _tokens(question)

        # Tables ranked in top-10 above the target (or just all top-10 if target absent)
        cutoff = (rank - 1) if rank and rank <= 10 else 10
        above = predicted[:cutoff]

        target_doc_overlap = len(query_toks & target_body_tokens) / max(len(query_toks), 1)
        target_schema_overlap = len(query_toks & schema_tokens.get(TARGET_TABLE, set())) / max(len(query_toks), 1)

        confounders = []
        for tid in above[:5]:
            body_toks = _tokens(table_body.get(tid, ''))
            doc_overlap = len(query_toks & body_toks) / max(len(query_toks), 1)
            s_overlap = len(query_toks & schema_tokens.get(tid, set())) / max(len(query_toks), 1)
            # Tokens that match query in confounder but NOT in target doc — why ColBERT prefers confounder
            doc_exclusive_matches = sorted(query_toks & body_toks - target_body_tokens)
            confounders.append({
                "table": tid,
                "doc_overlap": round(doc_overlap, 4),
                "schema_overlap": round(s_overlap, 4),
                "query_tokens_matching_only_confounder_doc": doc_exclusive_matches[:8],
            })

        rec = {
            "question": question,
            "hit_at_10": hit_at_10,
            "target_rank": rank,
            "target_doc_overlap": round(target_doc_overlap, 4),
            "target_schema_overlap": round(target_schema_overlap, 4),
            "confounders_above": confounders,
        }
        records.append(rec)

        status = f"HIT@{rank}" if hit_at_10 else ("HIT@" + str(rank) if rank else "MISS")
        print(f"\n[{status}] {question[:90]}")
        print(f"  {TARGET_TABLE}: doc_overlap={target_doc_overlap:.3f}  schema_overlap={target_schema_overlap:.3f}")
        for c in confounders:
            print(f"  > {c['table']:<15} doc={c['doc_overlap']:.3f}  schema={c['schema_overlap']:.3f}"
                  f"  only_in_confounder={c['query_tokens_matching_only_confounder_doc'][:5]}")

    misses = [r for r in records if not r['hit_at_10']]
    hits = [r for r in records if r['hit_at_10']]
    print(f"\n{'='*70}")
    print(f"Total {TARGET_TABLE} questions: {len(records)}")
    print(f"Hits@10: {len(hits)} / {len(records)}  ({len(misses)} misses)")
    if misses:
        avg_doc_gap = sum(
            (c['doc_overlap'] - r['target_doc_overlap'])
            for r in misses for c in r['confounders_above']
        ) / max(sum(len(r['confounders_above']) for r in misses), 1)
        avg_schema_gap = sum(
            (r['target_schema_overlap'] - c['schema_overlap'])
            for r in misses for c in r['confounders_above']
        ) / max(sum(len(r['confounders_above']) for r in misses), 1)
        print(f"Avg confounder doc_overlap advantage over target: {avg_doc_gap:+.4f}")
        print(f"Avg target schema_overlap advantage over confounders: {avg_schema_gap:+.4f}")
        print("(Positive schema_overlap gap = reranker would help; negative = wouldn't)")

    with open(args.output, 'w') as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
