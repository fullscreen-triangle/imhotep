#!/usr/bin/env python3
"""
Base validation class for all Imhotep framework validations.

This module provides the foundational validation infrastructure that all
specific validators inherit from, ensuring consistent methodology and results format.
"""

import json
import time
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime, timezone
import warnings


class ValidationBase(ABC):
    """
    Base class for all Imhotep framework validators.
    
    This class provides common infrastructure including:
    - Standardized validation methodology
    - Result formatting and storage
    - Statistical analysis utilities
    - Error handling and logging
    - Reproducibility controls
    """
    
    def __init__(self, 
                 name: str,
                 description: str,
                 random_seed: int = 42,
                 verbose: bool = True,
                 output_dir: Optional[str] = None):
        """
        Initialize base validator.
        
        Args:
            name: Name of the validation module
            description: Description of what this validator tests
            random_seed: Random seed for reproducibility
            verbose: Whether to print detailed progress information
            output_dir: Directory to save validation results
        """
        self.name = name
        self.description = description
        self.random_seed = random_seed
        self.verbose = verbose
        self.output_dir = Path(output_dir) if output_dir else Path("./validation_results")
        
        # Initialize logging
        self.logger = self._setup_logger()
        
        # Set random seeds for reproducibility
        self._set_random_seeds()
        
        # Validation metadata
        self.validation_metadata = {
            'validator_name': self.name,
            'validator_description': self.description,
            'validation_version': '1.0.0',
            'random_seed': self.random_seed,
            'start_time': None,
            'end_time': None,
            'duration_seconds': None,
            'python_version': None,
            'numpy_version': None,
            'system_info': self._get_system_info()
        }
        
        # Results storage
        self.results = {}
        self.detailed_results = {}
        self.statistical_analysis = {}
        
        if self.verbose:
            self.logger.info(f"🧠 Initialized {self.name} validator")
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logger for this validator."""
        logger = logging.getLogger(f"imhotep.validation.{self.name}")
        logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _set_random_seeds(self):
        """Set random seeds for reproducible results."""
        np.random.seed(self.random_seed)
        
        try:
            import torch
            torch.manual_seed(self.random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_seed)
        except ImportError:
            pass
        
        try:
            import tensorflow as tf
            tf.random.set_seed(self.random_seed)
        except ImportError:
            pass
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for validation metadata."""
        import platform
        import sys
        
        return {
            'platform': platform.platform(),
            'python_version': sys.version,
            'numpy_version': np.__version__,
            'cpu_count': None,  # Will be filled by specific implementations
            'memory_gb': None,  # Will be filled by specific implementations
        }
    
    @abstractmethod
    def validate(self) -> Dict[str, Any]:
        """
        Run the validation experiments.
        
        This method must be implemented by all validators and should return
        a dictionary containing validation results in the standardized format.
        
        Returns:
            Dictionary containing validation results with the following structure:
            {
                'validation_passed': bool,
                'confidence': float,
                'p_value': float,
                'effect_size': float,
                'claims_tested': int,
                'claims_validated': int,
                'detailed_results': dict,
                'statistical_analysis': dict,
                'errors': list,
                'warnings': list
            }
        """
        pass
    
    def run_validation(self, save_results: bool = True) -> Dict[str, Any]:
        """
        Run the complete validation process.
        
        Args:
            save_results: Whether to save results to file
            
        Returns:
            Complete validation results
        """
        self.logger.info(f"🚀 Starting validation for {self.name}")
        start_time = time.time()
        self.validation_metadata['start_time'] = datetime.now(timezone.utc).isoformat()
        
        try:
            # Run the validation
            results = self.validate()
            
            # Add metadata
            end_time = time.time()
            self.validation_metadata['end_time'] = datetime.now(timezone.utc).isoformat()
            self.validation_metadata['duration_seconds'] = end_time - start_time
            
            # Combine results with metadata
            complete_results = {
                'metadata': self.validation_metadata,
                'validation_results': results,
                'summary': self._generate_summary(results)
            }
            
            # Save results if requested
            if save_results:
                self._save_results(complete_results)
            
            # Log completion
            status = "✅ PASSED" if results.get('validation_passed', False) else "❌ FAILED"
            self.logger.info(
                f"{status} Validation completed in {end_time - start_time:.2f}s"
            )
            
            return complete_results
            
        except Exception as e:
            self.logger.error(f"❌ Validation failed with error: {str(e)}")
            error_results = {
                'metadata': self.validation_metadata,
                'validation_results': {
                    'validation_passed': False,
                    'error': str(e),
                    'error_type': type(e).__name__
                },
                'summary': {'status': 'ERROR', 'error': str(e)}
            }
            
            if save_results:
                self._save_results(error_results)
            
            raise
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of validation results."""
        return {
            'status': 'PASSED' if results.get('validation_passed', False) else 'FAILED',
            'confidence': results.get('confidence', 0.0),
            'p_value': results.get('p_value', 1.0),
            'effect_size': results.get('effect_size', 0.0),
            'claims_tested': results.get('claims_tested', 0),
            'claims_validated': results.get('claims_validated', 0),
            'validation_rate': (
                results.get('claims_validated', 0) / 
                max(results.get('claims_tested', 1), 1)
            ),
            'duration': self.validation_metadata.get('duration_seconds', 0)
        }
    
    def _save_results(self, results: Dict[str, Any]):
        """Save validation results to file."""
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.name}_validation_{timestamp}.json"
        filepath = self.output_dir / filename
        
        # Save results
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"💾 Results saved to {filepath}")
    
    def calculate_statistical_significance(self, 
                                         experimental_data: np.ndarray,
                                         control_data: np.ndarray,
                                         alpha: float = 0.001) -> Dict[str, Any]:
        """
        Calculate statistical significance between experimental and control groups.
        
        Args:
            experimental_data: Data from experimental condition
            control_data: Data from control condition  
            alpha: Significance threshold (default: 0.001 for high confidence)
            
        Returns:
            Statistical analysis results
        """
        from scipy import stats
        
        # Calculate basic statistics
        exp_mean = np.mean(experimental_data)
        exp_std = np.std(experimental_data, ddof=1)
        ctrl_mean = np.mean(control_data)
        ctrl_std = np.std(control_data, ddof=1)
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(experimental_data, control_data)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(experimental_data) - 1) * exp_std**2 + 
                             (len(control_data) - 1) * ctrl_std**2) / 
                            (len(experimental_data) + len(control_data) - 2))
        cohens_d = (exp_mean - ctrl_mean) / pooled_std
        
        # Determine significance
        is_significant = p_value < alpha
        
        # Effect size interpretation
        effect_size_interpretation = "small"
        if abs(cohens_d) >= 0.8:
            effect_size_interpretation = "large"
        elif abs(cohens_d) >= 0.5:
            effect_size_interpretation = "medium"
        
        return {
            'experimental_mean': exp_mean,
            'experimental_std': exp_std,
            'control_mean': ctrl_mean,
            'control_std': ctrl_std,
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': is_significant,
            'alpha': alpha,
            'effect_size_cohens_d': cohens_d,
            'effect_size_interpretation': effect_size_interpretation,
            'improvement_factor': exp_mean / ctrl_mean if ctrl_mean != 0 else float('inf'),
            'sample_sizes': {
                'experimental': len(experimental_data),
                'control': len(control_data)
            }
        }
    
    def validate_theoretical_prediction(self,
                                      theoretical_value: float,
                                      experimental_values: np.ndarray,
                                      tolerance: float = 0.05,
                                      confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Validate a theoretical prediction against experimental data.
        
        Args:
            theoretical_value: Theoretical prediction
            experimental_values: Experimental measurements
            tolerance: Acceptable relative error (default: 5%)
            confidence_level: Confidence level for validation (default: 95%)
            
        Returns:
            Validation results for theoretical prediction
        """
        exp_mean = np.mean(experimental_values)
        exp_std = np.std(experimental_values, ddof=1)
        exp_sem = exp_std / np.sqrt(len(experimental_values))
        
        # Calculate relative error
        relative_error = abs(exp_mean - theoretical_value) / abs(theoretical_value)
        
        # Confidence interval
        from scipy import stats
        dof = len(experimental_values) - 1
        t_critical = stats.t.ppf((1 + confidence_level) / 2, dof)
        ci_lower = exp_mean - t_critical * exp_sem
        ci_upper = exp_mean + t_critical * exp_sem
        
        # Check if theoretical value is within confidence interval
        within_ci = ci_lower <= theoretical_value <= ci_upper
        
        # Check if relative error is within tolerance
        within_tolerance = relative_error <= tolerance
        
        # Overall validation
        validation_passed = within_ci and within_tolerance
        
        return {
            'theoretical_value': theoretical_value,
            'experimental_mean': exp_mean,
            'experimental_std': exp_std,
            'experimental_sem': exp_sem,
            'relative_error': relative_error,
            'tolerance': tolerance,
            'within_tolerance': within_tolerance,
            'confidence_interval': (ci_lower, ci_upper),
            'within_confidence_interval': within_ci,
            'validation_passed': validation_passed,
            'confidence_level': confidence_level,
            'sample_size': len(experimental_values)
        }
    
    def measure_performance_improvement(self,
                                      baseline_performance: np.ndarray,
                                      improved_performance: np.ndarray) -> Dict[str, Any]:
        """
        Measure performance improvement over baseline.
        
        Args:
            baseline_performance: Baseline method performance
            improved_performance: Improved method performance
            
        Returns:
            Performance improvement analysis
        """
        baseline_mean = np.mean(baseline_performance)
        improved_mean = np.mean(improved_performance)
        
        # Calculate improvement metrics
        absolute_improvement = improved_mean - baseline_mean
        relative_improvement = absolute_improvement / baseline_mean if baseline_mean != 0 else 0
        improvement_factor = improved_mean / baseline_mean if baseline_mean != 0 else float('inf')
        
        # Statistical significance
        stats_result = self.calculate_statistical_significance(
            improved_performance, baseline_performance
        )
        
        return {
            'baseline_mean': baseline_mean,
            'improved_mean': improved_mean,
            'absolute_improvement': absolute_improvement,
            'relative_improvement': relative_improvement,
            'improvement_factor': improvement_factor,
            'percentage_improvement': relative_improvement * 100,
            'statistical_significance': stats_result,
            'is_significant_improvement': (
                stats_result['is_significant'] and 
                improvement_factor > 1.0
            )
        }
    
    def generate_validation_report(self) -> str:
        """Generate a human-readable validation report."""
        if not self.results:
            return "No validation results available. Run validation first."
        
        report = f"""
🧠 IMHOTEP VALIDATION REPORT: {self.name.upper()}
{'=' * 60}

Description: {self.description}

VALIDATION SUMMARY:
- Status: {'✅ PASSED' if self.results.get('validation_passed') else '❌ FAILED'}  
- Confidence: {self.results.get('confidence', 0):.3f}
- P-value: {self.results.get('p_value', 1):.6f}
- Effect Size: {self.results.get('effect_size', 0):.3f}
- Claims Tested: {self.results.get('claims_tested', 0)}
- Claims Validated: {self.results.get('claims_validated', 0)}

STATISTICAL ANALYSIS:
{self._format_statistical_results()}

DETAILED RESULTS:
{self._format_detailed_results()}

METADATA:
- Duration: {self.validation_metadata.get('duration_seconds', 0):.2f} seconds
- Random Seed: {self.validation_metadata.get('random_seed')}
- Timestamp: {self.validation_metadata.get('start_time', 'Unknown')}
"""
        return report
    
    def _format_statistical_results(self) -> str:
        """Format statistical analysis results for report."""
        if not self.statistical_analysis:
            return "No statistical analysis available."
        
        formatted = []
        for key, value in self.statistical_analysis.items():
            if isinstance(value, dict):
                formatted.append(f"- {key}:")
                for subkey, subvalue in value.items():
                    formatted.append(f"  - {subkey}: {subvalue}")
            else:
                formatted.append(f"- {key}: {value}")
        
        return "\n".join(formatted)
    
    def _format_detailed_results(self) -> str:
        """Format detailed results for report."""
        if not self.detailed_results:
            return "No detailed results available."
        
        formatted = []
        for key, value in self.detailed_results.items():
            formatted.append(f"- {key}: {value}")
        
        return "\n".join(formatted)
