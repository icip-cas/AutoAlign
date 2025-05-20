import hashlib
import json
import os
import time
from pathlib import Path


def maybe_load_evaluation_report(report_output_path: str | None) -> dict:
    """
    Load the evaluation report from the given path.
    If the file does not exist, return an empty dictionary.

    Args:
        report_output_path: Path to load the evaluation report.

    Returns:
        Evaluation report.
    """
    if report_output_path is None:
        return {}
    if os.path.exists(report_output_path):
        retry_count = 3
        while retry_count > 0:
            try:
                with open(report_output_path, "r") as f:
                    evaluation_report = json.load(f)
                break
            except json.decoder.JSONDecodeError:
                retry_count -= 1
                time.sleep(1)
    else:
        evaluation_report = {}
    return evaluation_report


def save_evaluation_report(report_output_path: str, evaluation_report: dict) -> None:
    """
    Save the evaluation report to the given path.

    Args:
        report_output_path: Path to save the evaluation report.

        evaluation_report: Evaluation report.
    """
    # mkdir if necessary
    report_output_path = os.path.abspath(report_output_path)
    os.makedirs(os.path.dirname(report_output_path), exist_ok=True)
    if os.path.exists(report_output_path):
        existing_report = maybe_load_evaluation_report(report_output_path)
        for k, v in existing_report.items():
            if k in evaluation_report:
                continue
            else:
                evaluation_report[k] = v

    with open(report_output_path, "w") as f:
        json.dump(evaluation_report, f, indent=4)

