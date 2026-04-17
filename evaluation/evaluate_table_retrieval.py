import argparse
import json
import numpy as np
import os
import pandas as pd
from itertools import chain
from matplotlib import pyplot as plt
from tqdm import tqdm
from typing import get_args, List, Dict
from time import perf_counter

import config
from models.retrievers.base_retriever import BaseRetriever
from utils.custom_types import QueryType
from utils.global_functions import load_dataset, load_model_from_path, parse_for_table_id


def evaluate_table_retrieval(
        model_path: str,
        dataset_path: str,
        query_type: QueryType,
        k: int = 5,
        model_kwargs: Dict = None) -> Dict:
    """
        Evaluates table retrieval performance of a given model on a given test dataset.

        :param model_path: path to the model
        :param dataset_path: path to the dataset file
        :param query_type: type of query to evaluate ('sexp', 'sql' or 'simplified_sql')
        :param k: The 'k' for accuracy@k calculation
        :param model_kwargs: optional keyword arguments for model instantiation
        :return: dictionary containing the used model, dataset, query_type and EM + Acc@K metrics
    """
    if model_kwargs is None:
        model_kwargs = {}
    dataset = load_dataset(dataset_path)
    model: BaseRetriever = load_model_from_path(model_path, **model_kwargs)

    exact_match = 0
    acc_at_k = 0
    acc_until_k = np.zeros(k)
    total = len(dataset)
    perf_time = []

    for item in tqdm(dataset, desc="Evaluating table retrieval", bar_format=config.TQDM_BAR_FMT):
        question = item.question
        answer = item[query_type]
        ground_truth_table = parse_for_table_id(answer, query_type)
        
        if not ground_truth_table:
            total -= 1
            continue

        s_t = perf_counter()
        predicted_tables = list(model.retrieve_tables(question, k=k).keys())
        perf_time.append(perf_counter() - s_t)

        # Exact Match (EM)
        # Check if all the first N tables of the predicted tables match the number N tables in the ground truth
        if set(predicted_tables[:len(ground_truth_table)]) == set(ground_truth_table):
            exact_match += 1
        
        # Accuracy@K
        # Check how many ground truth tables are in the first K tables predicted by the model
        acc_at_k += len(set(ground_truth_table) & set(predicted_tables[:k])) / len(ground_truth_table)

        matches = sorted(np.argwhere(np.isin(predicted_tables, list(ground_truth_table))).ravel())
        if len(matches) == 0:
            continue
        acc_until_k += np.array(list(chain.from_iterable(
            [[0.0] * min(k, matches[0])] +
            [[1 / len(matches) * (i + 1)] * ((matches[i + 1] if i + 1 < len(matches) else k) - p)
             for i, p in enumerate(matches)]
        )))

    em_accuracy = exact_match / total if total > 0 else 0
    acc_at_k = acc_at_k / total if total > 0 else 0
    acc_until_k = acc_until_k / total if total > 0 else 0

    results = {
        "model_path": model_path,
        "dataset_path": dataset_path,
        "query_type": query_type,
        "metrics": {
            "exact_match_accuracy": em_accuracy,
            f"accuracy_at_{k}": acc_at_k,
            "accuracies_until_k": list(acc_until_k),
            "mean_perf_time_in_seconds": float(np.mean(perf_time)),
            "median_perf_time_in_seconds": float(np.median(perf_time)),
        },
        "total_questions": total,
    }

    return results


def plot_graph(json_files: List[str], k: int):
    result_df = pd.DataFrame(index=range(k))
    for file in json_files:
        with open(file) as f:
            results = json.load(f)
            if 'accuracies_until_k' not in results.get('metrics', {}):
                continue

            result_df[results['model_path'].split('/')[-1].replace('.py', '')] = pd.Series(results['metrics']['accuracies_until_k'][:k])

    # Generate graphs
    result_df = result_df.reindex(sorted(result_df.columns), axis=1)
    range_x = len(result_df.index)
    x_axis = np.arange(1, range_x + 1, 1)
    plt.figure(figsize=(10, 7))
    for func in result_df:
        plt.plot(x_axis, result_df[func], label=func)

    max_tables = len(os.listdir(config.DB_ODATA3_FILES))

    plt.plot(x_axis, np.linspace(0, 1, max_tables)[:range_x], 'k--', label='Random')
    plt.xlim([min(x_axis), max(x_axis)])
    plt.ylim([0.0, 1.0])
    plt.xlabel("Top K retrieved tables")
    plt.ylabel("Accuracy")
    plt.title("accuracy@K for entity retrieval methods")
    plt.grid()
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()

    path = f"{os.path.dirname(args.output_path)}/er_results.png"
    plt.savefig(path)
    print(f"Saved graph to {path}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate table retrieval performance.", allow_abbrev=False)
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model.")
    parser.add_argument("--dataset_path", type=str, default=config.TEST_QA_FILE,
                        help="Path to the testing question-answer pairs set.")
    parser.add_argument("--query_type", type=str, choices=get_args(QueryType), default='sql',
                        help="Type of query to evaluate.")
    parser.add_argument("--k", "-k", type=int, default=5,
                        help="Value of K for accuracy@K.")
    parser.add_argument("--output_path", type=str,
                        default=f"evaluation/results/table_retrieval_{config.LANGUAGE}_results.json",
                        help="Path to save the results JSON file.")
    parser.add_argument("-p", "--plot-only", action="store_true",
                        help="Only plot the results of previous evaluations found in the directory of --output_path.")
    args, unknown = parser.parse_known_args()

    model_kwargs = {}
    i = 0
    while i < len(unknown):
        if unknown[i].startswith('--'):
            key = unknown[i].lstrip('-')
            if i + 1 < len(unknown) and not unknown[i + 1].startswith('-'):
                model_kwargs[key] = unknown[i + 1]
                i += 2
            else:
                model_kwargs[key] = True
                i += 1
        else:
            i += 1  # skip spurious tokens (e.g. ' ' from shell backslash-space escaping)

    if not args.plot_only:
        results = evaluate_table_retrieval(args.model_path, args.dataset_path, args.query_type, args.k,
                                           model_kwargs=model_kwargs)

        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=4)

        print(json.dumps(results, indent=4))
        print(f"Results saved to {args.output_path}")

    dir_name = os.path.dirname(args.output_path)
    print(f"Plotting results founds in {dir_name}")
    jsons = [f"{dir_name}/{file}" for file in os.listdir(dir_name) if '.json' in file]
    plot_graph(jsons, args.k)
