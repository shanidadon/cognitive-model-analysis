#!/usr/bin/env python3
"""CLI Overview - Comprehensive experiment management utilities.

This module provides utilities for running experiments across all BML Vision models,
managing experiment batches, and generating comparative analysis reports.

Usage:
    python -m src.experiment.cli_overview --config configs/cifar10.yaml --output results/all_models.csv
"""

import argparse
import csv
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, List

from .schemas import ExperimentConfig, Model, load_config
from .pipeline import ExperimentRidge

logger = logging.getLogger(__name__)


def append_to_model_accuracy_csv(model_name: str, train_accuracy=None, test_accuracy=None, error_message=None, update_existing=False):
    """Update or append model results to CSV with proper error handling.
    
    Args:
        model_name: Name of the model
        train_accuracy: Training accuracy (currently N/A for ridge regression)
        test_accuracy: Test accuracy or None for failures
        error_message: Error message if experiment failed
        update_existing: If True, update the most recent entry for this model
    """
    csv_path = Path("results/model_accuracy.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Read existing data if file exists
    existing_data = []
    fieldnames = ['Model Name', 'Train Accuracy', 'Test Accuracy', 'Timestamp', 'Error']
    
    if csv_path.exists():
        try:
            with open(csv_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                existing_data = list(reader)
        except Exception as e:
            logger.warning(f"Could not read existing CSV: {e}")
    
    # Find existing entry for this model (most recent)
    existing_entry_idx = None
    if update_existing:
        for i in range(len(existing_data) - 1, -1, -1):
            if existing_data[i].get('Model Name') == model_name:
                existing_entry_idx = i
                break
    
    # Create new entry
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if error_message:
        new_entry = {
            'Model Name': model_name,
            'Train Accuracy': 'FAILED',
            'Test Accuracy': 'FAILED', 
            'Timestamp': timestamp,
            'Error': error_message
        }
    else:
        new_entry = {
            'Model Name': model_name,
            'Train Accuracy': 'N/A' if train_accuracy is None else f"{train_accuracy:.4f}",
            'Test Accuracy': 'N/A' if test_accuracy is None else f"{test_accuracy:.4f}",
            'Timestamp': timestamp,
            'Error': ''
        }
    
    # Update existing entry or append new one
    if existing_entry_idx is not None and test_accuracy is not None:
        # Update the existing entry with results
        existing_data[existing_entry_idx].update(new_entry)
        logger.info(f"Updated existing entry for {model_name}")
    else:
        # Append new entry
        existing_data.append(new_entry)
        logger.info(f"Added new entry for {model_name}")
    
    # Write back to CSV
    try:
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_data)
        logger.info(f"Updated model_accuracy.csv for {model_name}")
    except Exception as e:
        logger.error(f"Failed to write CSV: {e}")


def get_available_model_names() -> List[str]:
    """Get all available model names from the Model schema."""
    return list(Model.model_fields["name"].annotation.__args__)


def run_experiment_for_model(base_config_path: str, model_name: str) -> Dict:
    """Run experiment for a single model with comprehensive error handling.
    
    Args:
        base_config_path: Path to base YAML configuration file
        model_name: Name of the model to run experiment with
        
    Returns:
        Dictionary with experiment results
    """
    logger.info(f"Starting experiment with model: {model_name}")
    
    try:
        # Load base configuration
        base_config = load_config(base_config_path)
        
        # Create modified config for this model
        config_dict = base_config.model_dump()
        config_dict['model']['name'] = model_name
        config_dict['experiment']['name'] = f"{config_dict['experiment']['name']}_{model_name.lower()}"
        
        # Create new config instance
        experiment_config = ExperimentConfig(**config_dict)
        
        # Log experiment start with placeholder entry
        append_to_model_accuracy_csv(
            model_name=model_name,
            test_accuracy=None  # Will be updated on success
        )
        
        # Initialize experiment with detailed error catching
        try:
            experiment = ExperimentRidge(config=experiment_config)
            logger.info(f"✓ ExperimentRidge initialized for {model_name}")
            
            # Run the complete experiment pipeline
            logger.info(f"Starting complete experiment pipeline for {model_name}...")
            full_results = experiment.run()
            
            # Extract test accuracy from nested results structure
            test_metrics = full_results.get('test_metrics', {})
            test_accuracy = test_metrics.get('test_accuracy')
            
            logger.info(f"✓ Experiment completed: accuracy={test_accuracy:.4f}")
            
            # Update CSV with success (update existing entry)
            append_to_model_accuracy_csv(
                model_name=model_name,
                test_accuracy=test_accuracy,
                update_existing=True
            )
            
            return {
                'model_name': model_name,
                'accuracy': test_accuracy,
                'f1_score': test_metrics.get('test_f1', None),
                'precision': test_metrics.get('test_precision', None),
                'recall': test_metrics.get('test_recall', None),
                'best_alpha': getattr(experiment, 'best_alpha', None),
                'status': 'success',
                'error': None
            }
            
        except Exception as exp_error:
            logger.error(f"✗ Experiment execution failed for {model_name}: {exp_error}")
            
            # Update CSV with detailed error (update existing entry)
            append_to_model_accuracy_csv(
                model_name=model_name,
                error_message=f"Experiment execution failed: {str(exp_error)}",
                update_existing=True
            )
            
            return {
                'model_name': model_name,
                'accuracy': None,
                'f1_score': None,
                'precision': None,
                'recall': None,
                'best_alpha': None,
                'status': 'experiment_failed',
                'error': str(exp_error)
            }
            
    except Exception as config_error:
        logger.error(f"✗ Configuration/initialization failed for {model_name}: {config_error}")
        
        # Add CSV entry with config error
        append_to_model_accuracy_csv(
            model_name=model_name,
            error_message=f"Configuration failed: {str(config_error)}"
        )
        
        return {
            'model_name': model_name,
            'accuracy': None,
            'f1_score': None,
            'precision': None,
            'recall': None,
            'best_alpha': None,
            'status': 'config_failed',
            'error': str(config_error)
        }


def run_all_models(base_config_path: str, output_csv: str = "results/all_models_results.csv"):
    """Run experiments on all available models and save results to CSV.
    
    Args:
        base_config_path: Path to base YAML configuration file
        output_csv: Path to output CSV file
        
    Returns:
        List of experiment results
    """
    
    # Get all model names from schema
    model_names = get_available_model_names()
    logger.info(f"Running experiments for {len(model_names)} models: {model_names}")
    
    results = []
    
    # Run experiment for each model
    for i, model_name in enumerate(model_names, 1):
        logger.info(f"[{i}/{len(model_names)}] Processing {model_name}")
        result = run_experiment_for_model(base_config_path, model_name)
        results.append(result)
    
    # Ensure output directory exists
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save results to CSV
    fieldnames = ['model_name', 'accuracy', 'f1_score', 'precision', 'recall', 'best_alpha', 'status', 'error']
    
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    logger.info(f"Results saved to {output_csv}")
    
    # Print summary
    successful_results = [r for r in results if r['status'] == 'success']
    failed_results = [r for r in results if r['status'] != 'success']
    
    print(f"\n{'='*80}")
    print(f"RIDGE REGRESSION EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"Total models: {len(model_names)}")
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {len(failed_results)}")
    
    if successful_results:
        # Sort by accuracy for better display
        successful_results.sort(key=lambda x: x['accuracy'] or 0, reverse=True)
        
        print(f"\n🏆 RESULTS (sorted by accuracy):")
        print(f"{'Model':<30} {'Accuracy':<10} {'F1-Score':<10} {'Best Alpha':<12}")
        print(f"{'-'*70}")
        
        for result in successful_results:
            accuracy = f"{result['accuracy']:.4f}" if result['accuracy'] is not None else "N/A"
            f1_score = f"{result['f1_score']:.4f}" if result['f1_score'] is not None else "N/A"
            best_alpha = f"{result['best_alpha']:.4f}" if result['best_alpha'] is not None else "N/A"
            print(f"{result['model_name']:<30} {accuracy:<10} {f1_score:<10} {best_alpha:<12}")
        
        best_model = successful_results[0]
        print(f"\n🥇 Best representation: {best_model['model_name']} (accuracy: {best_model['accuracy']:.4f})")
        print(f"   Best performing features for linear classification on CIFAR-10")
    
    if failed_results:
        print(f"\n❌ FAILED EXPERIMENTS:")
        for result in failed_results:
            print(f"  {result['model_name']}: {result['error']}")
    
    return results


def generate_comparison_report(results: List[Dict], output_path: str = "results/model_comparison_report.txt"):
    """Generate a detailed comparison report of model performance.
    
    Args:
        results: List of experiment results
        output_path: Path to save the report
    """
    successful_results = [r for r in results if r['status'] == 'success']
    successful_results.sort(key=lambda x: x['accuracy'] or 0, reverse=True)
    
    report_content = f"""
# Model Comparison Report - Ridge Regression on CIFAR-10

## Experiment Overview
This report compares the quality of features extracted from various pre-trained vision models
using ridge regression classification on CIFAR-10 dataset.

## Methodology
- Dataset: CIFAR-10 (50,000 training + 10,000 test images)
- Feature Extraction: Frozen pre-trained models as feature extractors
- Classification: Ridge regression with cross-validation
- Evaluation: Linear probing to assess representation quality

## Results Summary

Total Models Tested: {len(results)}
Successful Experiments: {len(successful_results)}
Failed Experiments: {len(results) - len(successful_results)}

## Detailed Results

"""
    
    if successful_results:
        report_content += "### Top Performing Models\n\n"
        for i, result in enumerate(successful_results[:10], 1):  # Top 10
            report_content += f"{i}. **{result['model_name']}**\n"
            report_content += f"   - Accuracy: {result['accuracy']:.4f}\n"
            report_content += f"   - F1-Score: {result['f1_score']:.4f}\n" if result['f1_score'] else ""
            report_content += f"   - Best Alpha: {result['best_alpha']:.4f}\n" if result['best_alpha'] else ""
            report_content += "\n"
        
        # Analysis section
        report_content += "### Analysis\n\n"
        best_model = successful_results[0]
        worst_model = successful_results[-1] if len(successful_results) > 1 else None
        
        report_content += f"- **Best performing model**: {best_model['model_name']} ({best_model['accuracy']:.4f} accuracy)\n"
        if worst_model:
            report_content += f"- **Lowest performing model**: {worst_model['model_name']} ({worst_model['accuracy']:.4f} accuracy)\n"
            improvement = ((best_model['accuracy'] - worst_model['accuracy']) / worst_model['accuracy']) * 100
            report_content += f"- **Performance gap**: {improvement:.1f}% improvement from worst to best\n"
        
        avg_accuracy = sum(r['accuracy'] for r in successful_results) / len(successful_results)
        report_content += f"- **Average accuracy**: {avg_accuracy:.4f}\n"
        
    # Save report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report_content)
    
    logger.info(f"Comparison report saved to {output_path}")


def main():
    """Main function to run all model experiments with CLI overview utilities."""
    parser = argparse.ArgumentParser(description="CLI Overview - Run experiments on all BML Vision models")
    parser.add_argument(
        "--config", 
        default="configs/cifar10.yaml",
        help="Base configuration file (default: configs/cifar10.yaml)"
    )
    parser.add_argument(
        "--output", 
        default="results/all_models_results.csv",
        help="Output CSV file (default: results/all_models_results.csv)"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate detailed comparison report"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Validate config file exists
    if not Path(args.config).exists():
        logger.error(f"Configuration file not found: {args.config}")
        return 1
    
    # Run experiments
    try:
        results = run_all_models(args.config, args.output)
        
        # Generate report if requested
        if args.report:
            report_path = Path(args.output).parent / "model_comparison_report.txt"
            generate_comparison_report(results, str(report_path))
        
        # Return exit code based on results
        successful_count = len([r for r in results if r['status'] == 'success'])
        if successful_count == 0:
            logger.error("No experiments completed successfully")
            return 1
        elif successful_count < len(results):
            logger.warning("Some experiments failed")
            return 2
        else:
            logger.info("All experiments completed successfully")
            return 0
            
    except Exception as e:
        logger.error(f"CLI overview script failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())