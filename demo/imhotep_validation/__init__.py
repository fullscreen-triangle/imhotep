#!/usr/bin/env python3
"""
Imhotep Framework Validation Package
====================================

Comprehensive experimental validation of all theoretical claims in the Imhotep neural architecture framework.

This package provides rigorous experimental validation for:
- Universal Problem-Solving Engine Theory
- BMD Information Catalysis Framework  
- Quantum Membrane Dynamics
- Visual Consciousness Framework
- Pharmaceutical Consciousness Optimization
- Self-Aware Neural Networks
- Production Performance Claims

Author: Kundai Farai Sachikonye
Institution: Independent Research Institute, Buhera, Zimbabwe
"""

__version__ = "1.0.0"
__author__ = "Kundai Farai Sachikonye"
__email__ = "kundai.sachikonye@wzw.tum.de"
__institution__ = "Independent Research Institute, Buhera, Zimbabwe"

# Core validation classes
from .core.comprehensive_validator import ComprehensiveValidator
from .core.quick_validator import QuickValidator
from .core.benchmark_suite import BenchmarkSuite
from .core.results_manager import ResultsManager

# Specific validation modules
from .validators.universal_problem_solving import UniversalProblemSolvingValidator
from .validators.bmd_information_catalysis import BMDInformationCatalysisValidator  
from .validators.quantum_membrane_dynamics import QuantumMembraneDynamicsValidator
from .validators.visual_consciousness import VisualConsciousnessValidator
from .validators.pharmaceutical_optimization import PharmaceuticalOptimizationValidator
from .validators.self_aware_networks import SelfAwareNetworksValidator
from .validators.production_performance import ProductionPerformanceValidator

# Utility modules
from .utils.statistical_analysis import StatisticalAnalyzer
from .utils.data_generator import DataGenerator
from .utils.visualization import ValidationVisualizer
from .utils.logging_utils import setup_validation_logging

# Configuration
from .config.validation_config import ValidationConfig
from .config.experimental_parameters import ExperimentalParameters

# Exception classes
class ImhotepValidationError(Exception):
    """Base exception for Imhotep validation errors."""
    pass

class ValidationFailureError(ImhotepValidationError):
    """Raised when a validation test fails."""
    pass

class InsufficientDataError(ImhotepValidationError):
    """Raised when insufficient data is available for validation."""
    pass

class ConfigurationError(ImhotepValidationError):
    """Raised when validation configuration is invalid."""
    pass

# Main validation functions
def validate_all_claims(output_dir="./results", detailed=True, save_json=True):
    """
    Validate all theoretical claims in the Imhotep framework.
    
    Args:
        output_dir (str): Directory to save validation results
        detailed (bool): Whether to run detailed validation tests
        save_json (bool): Whether to save results in JSON format
        
    Returns:
        dict: Comprehensive validation results
        
    Example:
        >>> import imhotep_validation as iv
        >>> results = iv.validate_all_claims()
        >>> print(f"Claims validated: {results['summary']['claims_validated']}")
    """
    validator = ComprehensiveValidator()
    return validator.validate_all_claims(
        output_dir=output_dir, 
        detailed=detailed, 
        save_json=save_json
    )

def quick_validation(components=None):
    """
    Run quick validation of core Imhotep components.
    
    Args:
        components (list): List of components to validate. 
                          If None, validates all core components.
                          
    Returns:
        dict: Quick validation results
        
    Example:
        >>> import imhotep_validation as iv
        >>> results = iv.quick_validation(['bmd', 'quantum'])
        >>> print(f"BMD validation: {results['bmd']['status']}")
    """
    validator = QuickValidator()
    return validator.validate_components(components)

def benchmark_performance(baselines=None, metrics=None):
    """
    Benchmark Imhotep performance against baseline methods.
    
    Args:
        baselines (list): Baseline methods to compare against
        metrics (list): Performance metrics to evaluate
        
    Returns:
        dict: Benchmark results showing performance comparisons
        
    Example:
        >>> import imhotep_validation as iv
        >>> benchmarks = iv.benchmark_performance(
        ...     baselines=['transformer', 'resnet'],
        ...     metrics=['accuracy', 'efficiency']
        ... )
        >>> print(f"Performance advantage: {benchmarks['improvement_factor']}×")
    """
    benchmark_suite = BenchmarkSuite()
    return benchmark_suite.run_benchmarks(baselines, metrics)

# Package metadata
VALIDATION_MODULES = [
    'universal_problem_solving',
    'bmd_information_catalysis', 
    'quantum_membrane_dynamics',
    'visual_consciousness',
    'pharmaceutical_optimization',
    'self_aware_networks',
    'production_performance'
]

SUPPORTED_METRICS = [
    'accuracy',
    'efficiency', 
    'information_density',
    'quantum_coherence',
    'consciousness_coherence',
    'processing_speed',
    'error_rate',
    'robustness'
]

BENCHMARK_BASELINES = [
    'classical_neural_network',
    'transformer',
    'resnet',
    'lstm',
    'conventional_quantum_processor',
    'standard_vision_system'
]

# Validation status tracking
_validation_status = {
    'initialized': False,
    'config_loaded': False,
    'last_validation': None,
    'total_validations': 0
}

def get_validation_status():
    """Get current validation package status."""
    return _validation_status.copy()

def initialize_validation_environment():
    """Initialize the validation environment with proper configuration."""
    global _validation_status
    
    # Set up logging
    setup_validation_logging()
    
    # Load configuration
    config = ValidationConfig()
    config.load_default_config()
    
    # Initialize random seeds for reproducibility  
    import numpy as np
    import torch
    import random
    
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    _validation_status['initialized'] = True
    _validation_status['config_loaded'] = True
    
    print("🧠 Imhotep Validation Environment Initialized")
    print(f"📊 Ready to validate {len(VALIDATION_MODULES)} theoretical frameworks")
    print(f"🎯 {len(SUPPORTED_METRICS)} performance metrics available")
    print(f"📈 {len(BENCHMARK_BASELINES)} baseline comparisons supported")

# Auto-initialize when package is imported
initialize_validation_environment()

__all__ = [
    # Core classes
    'ComprehensiveValidator',
    'QuickValidator', 
    'BenchmarkSuite',
    'ResultsManager',
    
    # Validators
    'UniversalProblemSolvingValidator',
    'BMDInformationCatalysisValidator',
    'QuantumMembraneDynamicsValidator', 
    'VisualConsciousnessValidator',
    'PharmaceuticalOptimizationValidator',
    'SelfAwareNetworksValidator',
    'ProductionPerformanceValidator',
    
    # Utilities
    'StatisticalAnalyzer',
    'DataGenerator',
    'ValidationVisualizer',
    'ValidationConfig',
    'ExperimentalParameters',
    
    # Main functions
    'validate_all_claims',
    'quick_validation',
    'benchmark_performance',
    'get_validation_status',
    'initialize_validation_environment',
    
    # Exceptions
    'ImhotepValidationError',
    'ValidationFailureError', 
    'InsufficientDataError',
    'ConfigurationError',
    
    # Constants
    'VALIDATION_MODULES',
    'SUPPORTED_METRICS',
    'BENCHMARK_BASELINES',
]
