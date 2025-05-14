import json
import pandas as pd

def calculate_task_statistics(file_path):
    data = []

    # Read the JSONL file
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Compute total trials per task
    total_trials = df.groupby("task")["num_demos"].sum().reset_index(name="total_trials")

    # Compute weighted success rate per task
    df["weighted_sr"] = df["num_demos"] * df["sr"]
    sr_per_task = df.groupby("task").agg(
        total_sr_sum=("weighted_sr", "sum"),
        total_demos=("num_demos", "sum")
    ).reset_index()
    
    # Calculate overall SR using weighted formula
    sr_per_task["overall_sr"] = sr_per_task["total_sr_sum"] / sr_per_task["total_demos"]

    # Merge total trials with calculated SR
    result = total_trials.merge(sr_per_task[["task", "overall_sr"]], on="task")

    # Calculate total SR across all tasks
    total_sr_all_tasks = df["weighted_sr"].sum() / df["num_demos"].sum()

    # Print results
    print("Success Rate Per Task:")
    print(result)
    print("\nTotal Success Rate Across All Tasks:", round(total_sr_all_tasks, 4))



# Example usage:
file_path = "/home/huser/robot-3dlotus/experiments/18-cos/preds/seed2025/results.jsonl"  # Replace with your actual file path
calculate_task_statistics(file_path)
