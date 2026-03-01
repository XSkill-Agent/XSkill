"""
Result writing, trajectory saving, and summary statistics utilities.
"""

import os
import json


def save_trajectory(save_directory, json_dict):
    """
    Save trajectory JSON to traj.jsonl.
    
    Args:
        save_directory: Directory to save files
        json_dict: Dictionary to append to traj.jsonl
    """
    os.makedirs(save_directory, exist_ok=True)
    with open(os.path.join(save_directory, "traj.jsonl"), "a+") as f:
        f.write(json.dumps(json_dict, ensure_ascii=False) + "\n")


def save_results(results, output_dir):
    """
    Save results to a JSONL file.
    
    Args:
        results: List of result dictionaries
        output_dir: Output directory path
    """
    output_file_path = os.path.join(output_dir, "results.jsonl")
    with open(output_file_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"Results saved to {output_file_path}")


def calculate_summary_metrics(results):
    """
    Calculate summary statistics from results.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Dictionary containing summary metrics
    """
    if not results:
        return {}
    
    num_samples = len(results)
    total_accuracy_score = sum(r.get('accuracy_score', 0) for r in results)
    average_accuracy = total_accuracy_score / num_samples if num_samples > 0 else 0

    # Average number of inference turns (count assistant messages)
    def _count_turns(r):
        ch = r.get('conversation_history', [])
        return sum(1 for m in ch if m.get('role') == 'assistant')
    
    average_turns = sum(_count_turns(r) for r in results) / num_samples if num_samples > 0 else 0

    summary_metrics = {
        "total_samples": num_samples,
        "overall_accuracy_score": round(average_accuracy, 4),
        "average_turns_per_sample": round(average_turns, 2),
    }

    return summary_metrics


def save_summary_metrics(summary_metrics, output_dir, print_message=False):
    """
    Save summary metrics to a JSON file.
    
    Args:
        summary_metrics: Dictionary of summary metrics
        output_dir: Output directory path
        print_message: Whether to print the save message (default: False)
    """
    summary_file_path = os.path.join(output_dir, "metrics_summary.json")
    with open(summary_file_path, 'w') as f:
        json.dump(summary_metrics, f, indent=4)
    
    if print_message:
        print(f"Summary metrics saved to: {summary_file_path}")


def print_summary(results, output_dir):
    """
    Calculate, save, and print summary statistics.
    This function matches the original behavior in infer_api.py:
    - Only processes if results is not empty
    - Prints summary first, then save message at the end
    
    Args:
        results: List of result dictionaries
        output_dir: Output directory path
    """
    if not results:
        return  # Original behavior: do nothing if empty, don't print message
    
    summary_metrics = calculate_summary_metrics(results)
    
    # Print summary first (matching original order)
    print("\n--- Evaluation Summary ---")
    print(f"Total samples processed: {summary_metrics['total_samples']}")
    print(f"Overall Accuracy Score: {summary_metrics['overall_accuracy_score']:.4f}")
    print(f"Average turns per sample: {summary_metrics['average_turns_per_sample']:.2f}")
    print(f"Summary metrics saved to: {os.path.join(output_dir, 'metrics_summary.json')}")
    print("------------------------\n")
    
    # Save summary metrics (without printing message, since we already printed it above)
    save_summary_metrics(summary_metrics, output_dir, print_message=False)

