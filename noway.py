import os
import json
import pandas as pd
from pathlib import Path

def extract_metrics_from_json(file_path):
    """
    Reads an eval_all_summary.json file and flattens its metrics into a dictionary 
    with tuple keys (Task, Subtask, Metric) to prepare for a multi-level Excel header.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    # Base identifier for the row
    row_data = {
        ('File Information', 'Metadata', 'File Path'): str(file_path),
        ('File Information', 'Metadata', 'Checkpoint'): data.get('checkpoint', 'N/A')
    }

    # Iterate through all tasks in the JSON
    for task_info in data.get('tasks', []):
        task_name = task_info.get('task', 'unknown_task')
        status = task_info.get('status', 'failed')
        
        # If task failed, log the status and move on
        if status != 'ok':
            row_data[(task_name, 'Overall', 'status')] = status
            continue

        raw = task_info.get('raw', {})
        
        # 1. Extract Overall Metrics
        # Look for the primary metrics attempted by the evaluator
        metric_keys_tried = task_info.get('metric_keys_tried', [])
        for key in metric_keys_tried:
            if key in raw and isinstance(raw[key], (int, float)):
                row_data[(task_name, 'Overall', key)] = raw[key]
                
        # Also grab standard top-level metrics that might not be in 'metric_keys_tried'
        for custom_key in ['glue_score', 'exact_match', 'f1', 'prompt_accuracy', 'instruction_accuracy']:
            if custom_key in raw and isinstance(raw[custom_key], (int, float)):
                row_data[(task_name, 'Overall', custom_key)] = raw[custom_key]

        # 2. Extract Sub-task Metrics
        # Different tasks organize their sub-metrics under different keys
        subtask_containers = ['per_subject', 'per_category', 'per_round', 'per_task']
        for container in subtask_containers:
            if container in raw:
                for subtask, metrics_dict in raw[container].items():
                    if isinstance(metrics_dict, dict):
                        for metric_name, metric_val in metrics_dict.items():
                            # Filter for numeric metrics to avoid dumping huge dictionaries/lists into cells
                            if isinstance(metric_val, (int, float)):
                                row_data[(task_name, subtask, metric_name)] = metric_val

    return row_data

def generate_excel_summary(root_directory, output_filename="evaluation_summary.xlsx"):
    """
    Finds all 'eval_all_summary.json' files, parses them, and saves a multi-index Excel file.
    """
    print(f"Scanning '{root_directory}' recursively for 'eval_all_summary.json' files...")
    
    # Recursively find all target JSON files
    json_files = list(Path(root_directory).rglob('eval_all_summary.json'))
    
    if not json_files:
        print("No 'eval_all_summary.json' files found in the specified directory.")
        return

    print(f"Found {len(json_files)} files. Extracting data...")
    all_rows = []
    for file_path in json_files:
        row_data = extract_metrics_from_json(file_path)
        if row_data:
            all_rows.append(row_data)

    if not all_rows:
        print("No valid data extracted from the found files.")
        return

    # Create DataFrame
    df = pd.DataFrame(all_rows)
    
    # Convert tuple column keys into a Pandas MultiIndex for hierarchical Excel headers
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=['Task', 'Subtask', 'Metric'])
    
    # Sort columns to group Tasks and Subtasks together neatly
    df = df.sort_index(axis=1)

    # Save to Excel
    print(f"Writing data to {output_filename}...")
    try:
        # Using context manager for ExcelWriter
        with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Metrics Summary')
            
            # Optional: Access the workbook and worksheet to freeze the header panes and first column
            worksheet = writer.sheets['Metrics Summary']
            worksheet.freeze_panes = 'D4' # Freezes the top 3 rows (headers) and first 3 columns (index/metadata)

        print(f"Success! Excel file generated: {output_filename}")
        
    except Exception as e:
        print(f"Failed to write Excel file. Error: {e}")

if __name__ == "__main__":
    # Define the target directory containing your run folders
    TARGET_DIRECTORY = "./"  # Update this to your base path, e.g., "/workspace/mnt/data_sda/..."
    
    # Define the desired output Excel file name
    OUTPUT_FILE = "eval_all_summary_compiled.xlsx"
    
    generate_excel_summary(TARGET_DIRECTORY, OUTPUT_FILE)
