import argparse
import duckdb
import json
import os
import pandas as pd
import sqlglot
import warnings
from time import time
from tqdm import tqdm
from typing import get_args, Tuple, Dict, Union, Literal, Optional

import config
from evaluation.metrics import (calculate_component_matching, lenient_execution_accuracy,
                                numeric_recall, record_accuracy, get_selection_metrics)
from models.generators.base_generator import BaseGenerator
from pipeline.db_executor import DBExecutor
from s_expression import Table, Expression
from s_expression.parser import parse, eval
from utils.answer_comparator import is_equal_frame
from utils.custom_types import QueryType, UnitCompatibilityError, FormatWarning, LLMResponse
from utils.global_functions import load_dataset, load_model_from_path, parse_for_table_id

if config.LANGUAGE == 'en':
    TEST_TABLES_PER_THEME = {
        'Education': ['84312ENG', '84732ENG', '81850ENG', '80393eng', '80509eng', '03753eng', '37931eng', '81491ENG', '80006eng'],
        'Energy': ['81154eng', '85799ENG', '81528ENG', '83376ENG', '83374ENG', '82117ENG', '82538ENG', '84918ENG', '84714ENG', '85899ENG', '82610ENG', '84917ENG', '85666ENG', '85592ENG', '80416ENG', '81567ENG', '82374ENG', '82375ENG', '82369ENG', '82371ENG', '83406ENG', '80101eng', '82379ENG', '71456eng', '70789eng', '71457eng', '83109ENG', '7516eng', '70802eng', '84672ENG', '37621eng', '83325ENG', '80099eng', '00377eng', '71840eng', '72002eng', '37281eng', '70846eng', '80324eng', '83141ENG', '81163eng', '00372eng', '83403ENG', '80100eng', '37215eng', '81606ENG'],
        'International trade': ['82659ENG', '82658ENG', '82616ENG'],
        'Manufacturing': ['7425eng', '81156eng', '81234eng', '85209ENG', '85806ENG', '85798ENG', '85770ENG', '83935ENG', '85771ENG', '83936ENG', '80274eng', '71835eng', '81238eng', '7055eng', '80092eng', '83838ENG', '81810ENG', '81984ENG', '81166eng', '81159eng', '83876ENG', '82444ENG', '37991eng', '37350ENG', '70696eng'],
        'Nature and environment': ['86242ENG', '86053ENG', '7477eng', '80408ENG', '82504ENG', '72002eng', '80370eng', '81010eng', '80447eng', '70946eng', '70947eng', '37221eng', '80448eng', '84735ENG', '83390ENG', '7063eng', '7467eng', '80138eng', '37687eng'],
        'Security and justice': ['82557ENG', '82558ENG', '82559ENG', '82522ENG', '82243ENG', '80045eng', '80426eng', '71482eng', '72007eng', '37957ENG', '37340eng', '37167eng', '37632eng', '37488eng', '37685ENG'],
    }
else:
    TEST_TABLES_PER_THEME = {
        'Energie': ['81156ned', '81154ned', '85799NED', '82538NED', '85677NED', '85337NED', '85080NED','84983NED', '84949NED', '84950NED', '86159NED', '85999NED', '85697NED', '85359NED', '85126NED', '84837NED', '84585NED', '84314NED', '83800NED', '83568NED', '83187NED', '83022NED', '83023NED', '83025NED', '83026NED', '81528NED', '85666NED', '85592NED', '80416ned', '84991NED', '81567NED', '85004NED', '70960ned', '86044NED', '85775NED', '85447NED', '85010NED', '84772NED', '84517NED', '84131NED', '85005NED', '82003NED', '70897NED', '81309NED', '37359', '81163ned', '70662ned', '71840ned', '70663ned', '70780ned', '72002ned', '70914ned', '84672NED', '7520', '7522', '37333', '37237', '37215', '7521', '7310SLEN', '07144', '07145', '37291', '7444', '83878NED', '83882NED', '7443', '37207', '7448', '82374NED', '7442', '7445', '82375NED', '7514', '7447', '82369NED', '7523', '82371NED', '7525', '7524', '7454', '70135ned', '80382ned', '83406NED', '80101ned', '82379NED', '70789ned', '83109NED', '7516', '71457ned', '71456ned', '82380NED', '70802ned', '84783NED', '84518NED', '84130NED', '70949ned', '71458ned', '37880klb', '71556ned'],
        'Industrie': ['81234ned', '85209NED', '80728ned', '81156ned', '85798NED', '7425zuiv', '85806NED', '85363NED', '83115NED', '85770NED', '83935NED', '85771NED', '83936NED', '81238ned', '71835ned', '80324ned', '37215', '7055', '37759', '70661NED', '70035NED', '81166ned', '71847ned', '70036NED', '81984NED', '80273ned', '83876NED', '83838NED', '81810NED', '37350', '37991pvr', '80111ned', '37154', '37569', '37971lix', '82113NED', '81460NED', '81975NED', '81974NED', '71933ned', '70753ned', '71250NED', '70037PRC'],
        'Internationale handel': ['82659NED', '82658NED'],
        'Natuur en milieu': ['37687wat', '70695NED', '80182ned', '7219WEER', '37368', '7041BBOD'],
        'Onderwijs': ['86223NED', '86087NED', '84973NED', '83861NED', '84732NED', '84312NED', '85356NED', '85353NED', '86052NED', '85740NED', '85372NED', '85051NED', '84773NED', '84947NED', '85702NED', '85525NED', '85701NED', '85489NED', '85490NED', '71478ned', '84780NED', '84274NED', '85907NED', '83966NED', '80393ned', '80509ned', '85453NED', '85313NED', '84337NED', '85834NED', '80384ned', '37220', '03753', '71517ned', '70896ned', '71562ned', '83873NED', '85511NED', '85820NED', '85214NED', '84972NED', '84696NED', '84455NED', '83969NED', '83663NED', '82859NED', '82252NED', '71822ned', '71229ned', '71042ned', '80171ned', '71043NED', '71890ned', '70637ned', '81869NED', '84311NED', '7523', '71796ned', '71825ned', '80468ned', '80284ned', '71493ned', '70134ned', '71450ned', '83295NED', '83296NED', '7265ONDW', '71294ned', '83393NED', '81788NED', '37846sol', '37746sol', '71202ned', '71247ned', '71199ned', '71073ned', '71074ned', '71201ned', '71200ned', '71054ned', '70962ned', '71172ned', '71113ned', '83894NED', '83893NED', '37379ou', '71535ned', '70901ned', '70222ned', '81491ned', '70903ned', '70902ned', '80197ned', '82299NED', '82298NED', '85184NED', '82816NED', '82275NED', '83324NED', '82123NED', '83613NED', '70762ned', '70761ned', '81473NED', '70630ned', '37182'],
        'Veiligheid en recht': ['82558NED', '82557NED', '82559NED', '37900']
    }

def execute_query(query: str, query_type: QueryType) -> Tuple[Union[str, Expression], pd.DataFrame, float]:
    """Execute the given query and calculate its execution time"""
    if query_type == 'sexp':
        t0 = time()
        query, answer = eval(parse(query), offline=True)
        execution_time = time() - t0
    elif query_type in ['sql', 'simplified_sql']:
        db = DBExecutor(
            tables=[Table(t) for t in parse_for_table_id(query, query_type)],
            measures=set(),
            dims=set()
        )
        simplified = query_type == 'simplified_sql'
        answer, _, execution_time = db.query_db(query, friendly_labels=False, simplified=simplified)
    else:
        raise ValueError(f"Unknown query type: {query_type}")

    return query, answer, execution_time


def get_question_type(query: str, query_type: QueryType) -> Optional[str]:
    """Determines the question type from an S-expression."""
    if query_type == 'sexp':
        # The order of bins matters for questions with multiple aggregations
        bins = ['PROP', 'SUM', 'AVG', 'MIN', 'MAX']

        has_join = 'JOIN' in query

        for agg in bins:
            if agg in query:
                if has_join:
                    return 'AGGJOIN'
                return agg

        if has_join:
            return 'JOIN'

        return 'VALUE'
    elif query_type in ['sql', 'simplified_sql']:
        try:
            expression = sqlglot.parse_one(query, read='duckdb')
        except Exception as e:
            return None

        # PROP
        if expression.find(sqlglot.exp.Div):
            return 'PROP'

        # JOIN / AGGJOIN
        is_join = bool(expression.find((
            sqlglot.exp.Join, sqlglot.exp.Union, sqlglot.exp.Intersect, sqlglot.exp.Except
        )))
        if is_join:
            is_aggjoin = any(
                isinstance(s, (sqlglot.exp.Sum, sqlglot.exp.Avg, sqlglot.exp.Min, sqlglot.exp.Max))
                for s in expression.selects
            )
            if is_aggjoin:
                return 'AGGJOIN'
            return 'JOIN'

        # Check aggregator in PIVOT to make distinction between VALUE and simple aggregators
        pivot = expression.find(sqlglot.exp.Pivot)
        if pivot:
            agg_expr = pivot.args.get('expressions')[0]
            has_groupby = bool(pivot.args.get('group'))

            # For MIN/MAX, having a group by indicates an ARGMIN/MAX operation
            if isinstance(agg_expr, sqlglot.exp.Min):
                return 'MIN' if has_groupby else 'VALUE'
            if isinstance(agg_expr, sqlglot.exp.Max):
                return 'MAX' if has_groupby else 'VALUE'

            # For SUM/AVG, if we pivot on one column, it's just selecting a value
            in_clause = pivot.args.get('in')
            num_pivot_values = len(in_clause.expressions) if in_clause else 0
            if isinstance(agg_expr, (sqlglot.exp.Sum, sqlglot.exp.Avg)):
                if num_pivot_values <= 1:
                    return 'VALUE'
                elif isinstance(agg_expr, sqlglot.exp.Sum):
                    return 'SUM'
                else:
                    return 'AVG'

        aggs = expression.find_all(sqlglot.exp.AggFunc)
        for agg_expr in aggs:
            if isinstance(agg_expr, sqlglot.exp.Sum):
                return 'SUM'
            if isinstance(agg_expr, sqlglot.exp.Avg):
                return 'AVG'
            if isinstance(agg_expr, sqlglot.exp.Min):
                return 'MIN'
            if isinstance(agg_expr, sqlglot.exp.Max):
                return 'MAX'

        return 'VALUE'
    else:
        raise ValueError(f"Unknown query type: {query_type}")


def evaluate_query_generation(
        model_path: str,
        dataset_path: str,
        output_path: str,
        query_type: QueryType,
        task: Literal['end-to-end', 'query-only'],
        model_kwargs: Dict = None,
        max_questions: int = None) -> Dict:
    """
        Evaluates query generation performance.

        :param model_path: path to the model
        :param dataset_path: path to the dataset file
        :param output_path: path to the generated queries from the model to. If a generated query is present for all
                            QA-paris in dataset_path, this can also be used for just re-calculating the metrics.
        :param query_type: type of query to evaluate ('sexp', 'sql' or 'simplified_sql')
        :param task: type of task to evaluate ('end-to-end' or 'query-only')
        :param model_kwargs: optional keyword arguments for model instantiation
        :return: dictionary containing the used model, dataset and metrics
     """
    if model_kwargs is None:
        model_kwargs = {}
    dataset = load_dataset(dataset_path)
    model: BaseGenerator = load_model_from_path(model_path, **model_kwargs)

    # Get previously generated results
    generated_answers = {}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            generated_answers = json.load(f)

    QUESTION_TYPES = ['VALUE', 'SUM', 'AVG', 'MIN', 'MAX', 'JOIN', 'PROP', 'AGGJOIN']

    input_token_count = 0
    output_token_count = 0
    exact_match = 0
    rec_accuracy = 0
    num_recall = 0
    execution_accuracy = {"strict": 0, "lenient": 0}
    component_scores = {'select_f1': 0.0, 'where_f1': 0.0, 'groupby_f1': 0.0, 'orderby_f1': 0.0, 'pivot_f1': 0.0}
    selection_scores = {
        "lenient": {'measure_f1': 0.0, 'dimension_f1': 0.0, 'observation_f1': 0.0},
        "strict": {'measure_f1': 0.0, 'dimension_f1': 0.0, 'observation_f1': 0.0}
    }
    error_scores = {
        'syntax_error': {
            'by_type': {q_type: 0 for q_type in QUESTION_TYPES},
            'by_theme': {theme: 0 for theme in TEST_TABLES_PER_THEME.keys()}
        },
        'formatting_error': {
            'by_type': {q_type: 0 for q_type in QUESTION_TYPES},
            'by_theme': {theme: 0 for theme in TEST_TABLES_PER_THEME.keys()}
        },
        'wrong_agg_func': {q_type: 0 for q_type in QUESTION_TYPES},
        'table_mismatch': 0,
        'unit_compatability_errors': 0,
        'extra_measures': 0, 'missing_measures': 0,
        'extra_dimensions': 0, 'missing_dimensions': 0
    }
    if query_type in ['sql', 'simplified_sql']:
        error_scores['missing_pivot'] = 0
    total = len(dataset)

    metrics_by_type = {
        q_type: {
            "count": 0,
            "exact_match": 0,
            "record_accuracy": 0,
            "numeric_recall": 0,
            "execution_accuracy": {"strict": 0, "lenient": 0},
            "selection_scores": {
                "lenient": {'measure_f1': 0.0, 'dimension_f1': 0.0, 'observation_f1': 0.0},
                "strict": {'measure_f1': 0.0, 'dimension_f1': 0.0, 'observation_f1': 0.0}
            }
        } for q_type in QUESTION_TYPES
    }

    questions_processed = 0
    for item in tqdm(dataset, desc=f"Evaluating query generation (Task: {task})", bar_format=config.TQDM_BAR_FMT):
        if max_questions is not None and questions_processed >= max_questions:
            total = questions_processed
            break
        questions_processed += 1
        question = item.question
        ground_truth_query = item[query_type]
        golden_tables = {t_id: {} for t_id in parse_for_table_id(ground_truth_query, query_type)}

        response = None
        if question in generated_answers:
            response = LLMResponse(**generated_answers[question])
            if response.query == "":
                response = None

        if response is None:
            response = model.generate_query(
                question,
                retrieved_tables=golden_tables if task == 'query-only' else None,
                query_type=query_type
            )

            generated_answers[question] = response.to_dict()
            with open(output_path, "w") as f:
                json.dump(generated_answers, f, indent=4)

        predicted_query = response.query
        input_token_count += response.input_token_count
        output_token_count += response.output_token_count

        if not predicted_query:
            total -= 1
            continue

        # Determine question type from s-expression
        q_type = get_question_type(item.sexp, query_type='sexp')
        metrics_by_type[q_type]["count"] += 1

        # Exact Match
        if query_type in ['sql', 'simplified_sql']:
            try:
                normalized_predicted = sqlglot.transpile(predicted_query, read='duckdb', pretty=False)[0]
                normalized_ground_truth = sqlglot.transpile(ground_truth_query, read='duckdb', pretty=False)[0]
            except Exception:
                normalized_predicted = ' '.join(predicted_query.split())
                normalized_ground_truth = ' '.join(ground_truth_query.split())
        else:  # sexp
            normalized_predicted = ' '.join(predicted_query.replace('(', ' ( ').replace(')', ' ) ').split())
            normalized_ground_truth = ' '.join(ground_truth_query.replace('(', ' ( ').replace(')', ' ) ').split())

        em_reward = 1 if normalized_predicted == normalized_ground_truth else 0
        exact_match += em_reward
        metrics_by_type[q_type]["exact_match"] += em_reward

        syntax_error = False
        rec_acc_reward = 0
        num_recall_reward = 0
        exec_acc_reward = 0
        lenient_exec_reward = 0
        try:
            # Check for query validity before execution
            if query_type == 'sexp':
                parse(predicted_query)
            elif query_type in ['sql', 'simplified_sql']:
                sqlglot.parse_one(predicted_query, read='duckdb')

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", FormatWarning)
                _, predicted_result, _ = execute_query(predicted_query, query_type)

                if any(issubclass(warn.category, FormatWarning) for warn in w):
                    # Formatting from retrieved data is wrong (usually indicates no measure was returned)
                    error_scores['formatting_error']['by_type'][q_type] += 1
                    for table_id in golden_tables:
                        for theme, ids in TEST_TABLES_PER_THEME.items():
                            if table_id in ids:
                                error_scores['formatting_error']['by_theme'][theme] += 1
                                break

            _, ground_truth_result, _ = execute_query(ground_truth_query, query_type)

            # Record Accuracy
            rec_acc_reward = record_accuracy(ground_truth_result, predicted_result)

            # Numeric Recall
            num_recall_reward = numeric_recall(ground_truth_result, predicted_result)

            # Execution Accuracy
            # Checking frame equality clogs the system memory for enormous tables. Trust me, the EX is 0 if you return more than 1 000 rows
            if len(predicted_result) < 1000 and is_equal_frame(ground_truth_result, predicted_result):
                exec_acc_reward = 1

            if len(predicted_result) < 1000:
                lenient_exec_reward = lenient_execution_accuracy(ground_truth_result, predicted_result)
        except UnitCompatibilityError:
            # Units are not compatible
            error_scores['unit_compatability_errors'] += 1
        except duckdb.IOException:
            # Table not found, will be added to the table_mismatch later
            pass
        except (duckdb.BinderException, duckdb.ParserException, sqlglot.errors.SqlglotError, SyntaxError) as e:
            # Syntactical errors
            syntax_error = True
            error_scores['syntax_error']['by_type'][q_type] += 1
            for table_id in golden_tables:
                for theme, ids in TEST_TABLES_PER_THEME.items():
                    if table_id in ids:
                        error_scores['syntax_error']['by_theme'][theme] += 1
                        break
        except (RuntimeError, Exception) as e:
            print(f"Caught unexpected error: {e}")

        rec_accuracy += rec_acc_reward
        metrics_by_type[q_type]['record_accuracy'] += rec_acc_reward

        num_recall += num_recall_reward
        metrics_by_type[q_type]['numeric_recall'] += num_recall_reward

        execution_accuracy['strict'] += exec_acc_reward
        metrics_by_type[q_type]['execution_accuracy']['strict'] += exec_acc_reward

        execution_accuracy['lenient'] += lenient_exec_reward
        metrics_by_type[q_type]['execution_accuracy']['lenient'] += lenient_exec_reward

        # Component Matching (SQL only)
        if query_type in ['sql', 'simplified_sql']:
            comp_scores = calculate_component_matching(predicted_query, ground_truth_query)
            for key, value in comp_scores.items():
                component_scores[key] += value

        # Selection-based F1 scores (lenient: always calculated, strict: zero on syntax error)
        sel_scores, sel_errors = get_selection_metrics(predicted_query, ground_truth_query, query_type)
        for key, value in sel_scores.items():
            selection_scores['lenient'][key] += value
            metrics_by_type[q_type]['selection_scores']['lenient'][key] += value
            if not syntax_error:
                selection_scores['strict'][key] += value
                metrics_by_type[q_type]['selection_scores']['strict'][key] += value

        error_scores['extra_measures'] += 1 if sel_errors['extra_measures'] > 0 else 0
        error_scores['missing_measures'] += 1 if sel_errors['missing_measures'] > 0 else 0
        error_scores['extra_dimensions'] += 1 if sel_errors['extra_dimensions'] > 0 else 0
        error_scores['missing_dimensions'] += 1 if sel_errors['missing_dimensions'] > 0 else 0

        # Error analysis for queries with correct syntax
        if not syntax_error:
            predicted_tables = parse_for_table_id(predicted_query, query_type)
            if set(golden_tables) != set(predicted_tables):
                error_scores['table_mismatch'] += 1

            if query_type in ['sql', 'simplified_sql'] and 'PIVOT' in ground_truth_query and 'PIVOT' not in predicted_query:
                error_scores['missing_pivot'] += 1

            # Determine if the model attempted to answer the questions using the correct 'type' of query
            predicted_agg_func = get_question_type(predicted_query, query_type=query_type)
            if (exec_acc_reward != 1. and rec_acc_reward != 1. and sel_scores['observation_f1'] != 1. and
                    predicted_agg_func is not None and q_type != predicted_agg_func):
                error_scores['wrong_agg_func'][q_type] += 1

    avg_input_token_count = input_token_count / total if total > 0 else 0
    avg_output_token_count = output_token_count / total if total > 0 else 0
    em_accuracy = exact_match / total if total > 0 else 0
    rec_accuracy = (rec_accuracy / total) if total > 0 else 0
    num_recall_accuracy = (num_recall / total) if total > 0 else 0
    ex_accuracy = {
        "strict": (execution_accuracy['strict'] / total) if total > 0 else 0,
        "lenient": (execution_accuracy['lenient'] / total) if total > 0 else 0,
    }
    avg_selection_scores = {
        "strict": {key: value / total for key, value in selection_scores['strict'].items()},
        "lenient": {key: value / total for key, value in selection_scores['lenient'].items()}
    }

    # Calculate average scores for each question type
    avg_metrics_by_type = {}
    for q_type, metrics in metrics_by_type.items():
        count = metrics["count"]
        if count > 0:
            avg_metrics_by_type[q_type] = {
                "exact_match_accuracy": metrics["exact_match"] / count,
                "record_accuracy": metrics["record_accuracy"] / count,
                "numeric_recall": metrics["numeric_recall"] / count,
                "execution_accuracy": {
                    "strict": metrics['execution_accuracy']['strict'] / count,
                    "lenient": metrics['execution_accuracy']['lenient'] / count
                },
                "selection_metrics": {
                    "lenient": {key: value / count for key, value in metrics['selection_scores']['lenient'].items()},
                    "strict": {key: value / count for key, value in metrics['selection_scores']['strict'].items()}
                },
                "total_questions": count
            }

    results = {
        "model_path": model_path,
        "dataset_path": dataset_path,
        "task": task,
        "query_type": query_type,
        "avg_input_token_count": avg_input_token_count,
        "avg_output_token_count": avg_output_token_count,
        "metrics": {
            "exact_match_accuracy": em_accuracy,
            "record_accuracy": rec_accuracy,
            "numeric_recall": num_recall_accuracy,
            "execution_accuracy": ex_accuracy,
            "selection_metrics": avg_selection_scores,
        },
        "error_analysis": error_scores,
        "metrics_by_question_type": avg_metrics_by_type,
        "total_questions": total
    }

    if query_type in ['sql', 'simplified_sql']:
        avg_component_scores = {key: value / total for key, value in component_scores.items()}
        results["metrics"]["component_matching"] = avg_component_scores

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate query generation performance.", allow_abbrev=False)
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--dataset_path", type=str, default=config.TEST_QA_FILE,
                        help="Path to the testing question-answer pairs set.")
    parser.add_argument("--query_type", type=str, choices=get_args(QueryType), default='sql',
                        help="Type of query to evaluate.")
    parser.add_argument("--task", type=str, choices=['end-to-end', 'query-only'], default='end-to-end',
                        help="Type task to evaluate. End-to-end included table retrieval. For query only the golden "
                             "tables will be provided to the model.")
    parser.add_argument("--output_path", type=str, default="evaluation/generated_queries.json",
                        help="Path to save the generated queries to.")
    parser.add_argument("--results_path", type=str, default="evaluation/query_generation_results.json",
                        help="Path to where to save the results JSON file with the metrics.")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Cap evaluation at N questions for cost estimation. Omit for full run.")
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
    results = evaluate_query_generation(args.model_path, args.dataset_path, args.output_path,
                                        args.query_type, args.task,
                                        model_kwargs=model_kwargs,
                                        max_questions=args.max_questions)

    os.makedirs(os.path.dirname(args.results_path), exist_ok=True)
    with open(args.results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Queries saved to {args.output_path}.")
    print(f"Results saved to {args.results_path}")
    print(json.dumps(results, indent=4))

    if args.max_questions:
        full_n = len(load_dataset(args.dataset_path))
        scale = full_n / args.max_questions
        est_input = results["avg_input_token_count"] * full_n
        est_output = results["avg_output_token_count"] * full_n
        # gpt-5-mini pricing: $0.25/1M input tokens, $2.00/1M output tokens
        est_cost = (est_input / 1e6 * 0.25) + (est_output / 1e6 * 2.00)
        print(f"\n--- Cost estimate for full {full_n} questions ---")
        print(f"  Avg input tokens/question : {results['avg_input_token_count']:.0f}")
        print(f"  Avg output tokens/question: {results['avg_output_token_count']:.0f}")
        print(f"  Estimated total input     : {est_input:,.0f} tokens")
        print(f"  Estimated total output    : {est_output:,.0f} tokens")
        print(f"  Estimated cost (gpt-5-mini): ~${est_cost:.2f}")
