"""
Process and aggregate evaluation results from multiple model runs and upload them to the HuggingFace Hub.

This script handles:
- Extracting results from JSON files
- Processing results into a structured format
- Uploading aggregated results to HuggingFace Hub (optional)
- Logging results to console
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
from datasets import Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_results(eval_results: Dict) -> Dict:
    """
    Extract relevant scores from evaluation results dictionary.

    Args:
        eval_results: Dictionary containing model evaluation results

    Returns:
        Dictionary containing model configuration and task scores
    """
    try:
        model_results = eval_results["config_general"]
        for task_name, task_score in eval_results["results"]["all"].items():
            model_results[task_name] = task_score
        return model_results
    except KeyError as e:
        logger.error(f"Missing required key in evaluation results: {e}")
        raise


def get_results_from_dir(results_dir: Path) -> List[Dict]:
    """
    Recursively process all result files from the given directory.

    Args:
        results_dir: Path to directory containing evaluation results

    Returns:
        List of processed result dictionaries
    """
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    results = []
    try:
        for author_dir in results_dir.iterdir():
            if not author_dir.is_dir():
                continue

            for model_dir in author_dir.iterdir():
                if not model_dir.is_dir():
                    continue

                for file in model_dir.iterdir():
                    if not file.suffix == ".json":
                        continue

                    try:
                        results.append(process_result_file(file))
                    except Exception as e:
                        logger.error(f"Error processing file {file}: {e}")
                        continue

        if not results:
            logger.warning("No valid result files found in the specified directory")

        return results

    except Exception as e:
        logger.error(f"Error reading results directory: {e}")
        raise


def process_result_file(file_path: Path) -> Dict:
    """
    Process a single result file.

    Args:
        file_path: Path to the result file

    Returns:
        Processed result dictionary
    """
    try:
        with file_path.open() as f:
            results = json.load(f)
        return extract_results(results)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in file: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        raise


def push_results_to_hub(model_results: List[Dict], repo_id: str) -> None:
    """
    Upload processed results to HuggingFace Hub.

    Args:
        model_results: List of processed result dictionaries
        repo_id: HuggingFace Hub repository ID to upload to
    """
    try:
        dataset = Dataset.from_list(model_results)
        dataset.push_to_hub(repo_id=repo_id)
        logger.info(f"Successfully pushed results to {repo_id}")
    except Exception as e:
        logger.error(f"Error pushing results to hub: {e}")
        raise


def display_results(results: List[Dict]) -> None:
    """
    Display results as a formatted table.

    Args:
        results: List of processed result dictionaries
    """
    try:
        df = pd.DataFrame(results)
        logger.info("\nResults Summary:")
        logger.info("\n" + str(df))

        # Log some basic statistics
        logger.info("\nSummary Statistics:")
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        logger.info("\n" + str(df[numeric_cols].describe()))

    except Exception as e:
        logger.error(f"Error displaying results: {e}")
        raise


def main(results_dir: str, repo_id: str = None) -> None:
    """
    Main function to process results and optionally upload to HuggingFace Hub.

    Args:
        results_dir: Directory containing evaluation results
        repo_id: Optional HuggingFace Hub repository ID to upload results to
    """
    try:
        results_path = Path(results_dir)
        results = get_results_from_dir(results_path)

        display_results(results)

        if repo_id:
            push_results_to_hub(results, repo_id)

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process model evaluation results and optionally upload to HuggingFace Hub"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing evaluation results",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        help="HuggingFace Hub repository ID to upload results to (optional)",
    )

    args = parser.parse_args()
    main(args.results_dir, args.repo_id)
