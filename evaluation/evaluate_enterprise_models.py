"""
Batch evaluator for pre-generated enterprise model outputs.
Reads enterprise model JSON files (question -> {query, tokens}) and runs
evaluation metrics without invoking a live model.

Usage:
    env PYTHONPATH=. LANGUAGE=en python3 evaluation/evaluate_enterprise_models.py
    env PYTHONPATH=. LANGUAGE=nl python3 evaluation/evaluate_enterprise_models.py
"""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import config

ENTERPRISE_DIR = "evaluation/results/enterprise_models"
lang = config.LANGUAGE


def main():
    files = sorted(
        p for p in Path(ENTERPRISE_DIR).glob("*.json")
        if f"_{lang}.json" in p.name or f"_{lang}_reasoning.json" in p.name
        if not p.name.endswith("_results.json")
    )

    if not files:
        print(f"No enterprise model files found for LANGUAGE={lang} in {ENTERPRISE_DIR}")
        sys.exit(1)

    print(f"Found {len(files)} file(s) to evaluate for LANGUAGE={lang}:")
    for f in files:
        print(f"  {f}")

    from evaluation.evaluate_query_generation import evaluate_query_generation
    from utils.global_functions import load_dataset

    dataset = load_dataset(config.TEST_QA_FILE)
    all_questions = [item.question for item in dataset]

    for model_file in files:
        results_path = str(model_file).replace(".json", "_results.json")
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_file.name}")
        print(f"Results  -> {results_path}")

        with open(model_file) as f:
            pre_generated = json.load(f)

        missing = [q for q in all_questions if q not in pre_generated]
        if missing:
            print(f"  Warning: {len(missing)} question(s) not in output file — treating as empty query (failure)")

        # Build a complete answers dict. Empty string re-triggers model generation,
        # so replace empty/missing queries with a comment — counts as syntax error/failure.
        PLACEHOLDER = "-- NOT GENERATED"
        complete_answers = {}
        for q, v in pre_generated.items():
            entry = dict(v)
            if not entry.get("query", "").strip():
                entry["query"] = PLACEHOLDER
            complete_answers[q] = entry
        for q in missing:
            complete_answers[q] = {"query": PLACEHOLDER, "input_token_count": 0, "output_token_count": 0}

        # Write to a temp file so the evaluator never overwrites the original
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(complete_answers, tmp, indent=4)
            tmp_path = tmp.name

        dummy_model = MagicMock()
        dummy_model.generate_query.side_effect = RuntimeError("Should never be called")

        try:
            with patch("evaluation.evaluate_query_generation.load_model_from_path", return_value=dummy_model):
                results = evaluate_query_generation(
                    model_path=str(model_file),  # unused (patched out)
                    dataset_path=config.TEST_QA_FILE,
                    output_path=tmp_path,         # temp copy with all questions filled
                    query_type="sql",
                    task="query-only",
                )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)

        metrics = results.get("metrics", {})
        sel = metrics.get("selection_metrics", {})
        print(f"  exec_acc (strict/lenient): "
              f"{metrics.get('execution_accuracy', {}).get('strict', 0):.3f} / "
              f"{metrics.get('execution_accuracy', {}).get('lenient', 0):.3f}")
        for mode in ("strict", "lenient"):
            s = sel.get(mode, {})
            print(f"  [{mode}] msrF1={s.get('measure_f1', 0):.3f}  "
                  f"dimF1={s.get('dimension_f1', 0):.3f}  "
                  f"obsF1={s.get('observation_f1', 0):.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
