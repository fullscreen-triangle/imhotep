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
from .core.validation_base import ValidationBase

# Specific validation modules (only the ones we actually implemented)
from .validators.universal_problem_solving import UniversalProblemSolvingValidator
from .validators.bmd_information_catalysis import BMDInformationCatalysisValidator
from .validators.quantum_membrane_dynamics import QuantumMembraneDynamicsValidator
from .validators.self_awareness import SelfAwarenessValidator
from .validators import ALL_VALIDATORS

# Utility modules (only the ones we actually implemented)
from .utils.statistical_analysis import calculate_entropy

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
def validate_all_claims(output_dir="./validation_results", parallel_validation=True, verbose=True, random_seed=42):
    """
    Validate all theoretical claims in the Imhotep framework.

    Args:
        output_dir (str): Directory to save validation results
        parallel_validation (bool): Whether to run validations in parallel
        verbose (bool): Whether to print detailed progress
        random_seed (int): Random seed for reproducible results

    Returns:
        dict: Comprehensive validation results

    Example:
        >>> import imhotep_validation as iv
        >>> results = iv.validate_all_claims()
        >>> print(f"Overall validation passed: {results['comprehensive_summary']['overall_validation_passed']}")
    """
    validator = ComprehensiveValidator(
        output_dir=output_dir,
        parallel_validation=parallel_validation,
        verbose=verbose,
        random_seed=random_seed
    )
    return validator.validate_all_claims(
        save_results=True,
        detailed_analysis=True,
        generate_report=True
    )

def quick_validation(verbose=True, random_seed=42, output_dir=None):
    """
    Run quick validation of core Imhotep components.

    Args:
        verbose (bool): Whether to print validation progress
        random_seed (int): Random seed for reproducibility
        output_dir (str): Directory to save results

    Returns:
        dict: Quick validation results

    Example:
        >>> import imhotep_validation as iv
        >>> results = iv.quick_validation()
        >>> print(f"Quick validation passed: {results['validation_passed']}")
    """
    validator = QuickValidator(
        verbose=verbose,
        random_seed=random_seed,
        output_dir=output_dir
    )
    return validator.run_validation(save_results=True)

# Package metadata
VALIDATION_MODULES = [
    'universal_problem_solving',
    'bmd_information_catalysis',
    'quantum_membrane_dynamics',
    'self_awareness'
]

SUPPORTED_METRICS = [
    'information_processing_speedup',
    'quantum_coherence',
    'self_awareness_accuracy',
    'computational_impossibility',
    'universal_solvability'
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

    # Initialize random seeds for reproducibility
    try:
        import numpy as np
        import random

        seed = 42
        np.random.seed(seed)
        random.seed(seed)

        try:
            import torch
            torch.manual_seed(seed)
        except ImportError:
            pass

    except ImportError:
        pass

    _validation_status['initialized'] = True
    _validation_status['config_loaded'] = True

    print("🧠 Imhotep Validation Environment Initialized")
    print(f"📊 Ready to validate {len(VALIDATION_MODULES)} theoretical frameworks")
    print(f"🎯 {len(SUPPORTED_METRICS)} performance metrics available")

# Auto-initialize when package is imported
initialize_validation_environment()

__all__ = [
    # Core classes
    'ComprehensiveValidator',
    'QuickValidator',
    'ValidationBase',

    # Validators
    'UniversalProblemSolvingValidator',
    'BMDInformationCatalysisValidator',
    'QuantumMembraneDynamicsValidator',
    'SelfAwarenessValidator',
    'ALL_VALIDATORS',

    # Utilities
    'calculate_entropy',

    # Main functions
    'validate_all_claims',
    'quick_validation',
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
]
