import os
import json
import csv
import regex as re
from operator import itemgetter

def summarize_results():
    """
    Summarizes the results from the query-only and end-to-end evaluation files.
    """
    result_dirs = {
        './results/query-only': 'query-only',
        './results/end-to-end': 'end-to-end',
    }
    output_csv = 'summary.csv'

    summary_data = []

    for results_dir, task_name in result_dirs.items():
        # Get all json files
        files_to_process = [f for f in os.listdir(results_dir) if f.endswith('.json')]

        for filename in files_to_process:
            # Extract metadata from filename
            parts = filename.replace('.json', '').split('_')
            model_name = parts[0]
            query_type = 'simplified_sql' if 'simplified_sql' in filename else 'sql'
            language = 'en' if re.search(fr"_en[\._]", filename) else 'nl'
            reasoning = 'reasoning' if 'reasoning' in parts else ''

            filepath = os.path.join(results_dir, filename)

            with open(filepath, 'r') as f:
                data = json.load(f)

                metrics = data.get('metrics', {})
                selection_metrics = metrics.get('selection_metrics', {})
                error_analysis = data.get('error_analysis', {})
                wrong_agg_func = error_analysis.get('wrong_agg_func', {})

                row = {
                    'task': task_name,
                    'model_name': model_name,
                    'language': language,
                    'query_type': query_type,
                    'reasoning': reasoning,
                    'EX (strict)': round(metrics['execution_accuracy'].get('strict'), 2),
                    'EX (lenient)': round(metrics['execution_accuracy'].get('lenient'), 2),
                    'RX': round(metrics.get('record_accuracy'), 2),
                    'NR': round(metrics.get('numeric_recall'), 2),
                    'msrF1 (strict)': round(selection_metrics['strict'].get('measure_f1'), 2),
                    'msrF1 (lenient)': round(selection_metrics['lenient'].get('measure_f1'), 2),
                    'dimF1 (strict)': round(selection_metrics['strict'].get('dimension_f1'), 2),
                    'dimF1 (lenient)': round(selection_metrics['lenient'].get('dimension_f1'), 2),
                    'obsF1 (strict)': round(selection_metrics['strict'].get('observation_f1'), 2),
                    'obsF1 (lenient)': round(selection_metrics['lenient'].get('observation_f1'), 2),
                    'input_tokens': round(data.get('avg_input_token_count'), 2),
                    'output_tokens': round(data.get('avg_output_token_count'), 2),
                    **wrong_agg_func
                }
                summary_data.append(row)
            
    if summary_data:
        # Sort the data
        summary_data.sort(key=itemgetter('task', 'language', 'query_type', 'model_name'))

        # Define header order
        header = [
            'task', 'language', 'model_name', 'query_type', 'reasoning',
            'EX (strict)', 'EX (lenient)', 'RX', 'NR',
            'msrF1 (strict)', 'dimF1 (strict)', 'obsF1 (strict)',
            'msrF1 (lenient)', 'dimF1 (lenient)', 'obsF1 (lenient)',
            'VALUE', 'SUM', 'AVG', 'MIN', 'MAX', 'JOIN', 'PROP', 'AGGJOIN',
            'input_tokens', 'output_tokens'
        ]
        
        # Add the rest of the keys (wrong_agg_func keys)
        all_keys = set()
        for row in summary_data:
            all_keys.update(row.keys())
            
        # Sort the remaining keys for consistent order
        remaining_keys = sorted([key for key in all_keys if key not in header])
        header.extend(remaining_keys)
        
        # Write to CSV
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()
            writer.writerows(summary_data)
        
        print(f"Successfully created {output_csv}")

if __name__ == '__main__':
    summarize_results()