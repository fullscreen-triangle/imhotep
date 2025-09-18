# Imhotep Validation Package - Installation Guide

This guide provides step-by-step instructions for installing and using the Imhotep Framework Validation Package.

## Prerequisites

- Python 3.7 or higher
- pip (Python package manager)
- Virtual environment (recommended)

## Installation

### 1. Set up a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv imhotep_env

# Activate virtual environment
# On Windows:
imhotep_env\Scripts\activate
# On macOS/Linux:
source imhotep_env/bin/activate
```

### 2. Install the Package

#### Option A: Development Installation (from source)
```bash
# Navigate to the demo directory containing setup.py
cd path/to/demo

# Install in development mode
pip install -e .
```

#### Option B: Standard Installation (when published)
```bash
pip install imhotep-validation
```

### 3. Verify Installation

```python
import imhotep_validation as iv
print(f"Imhotep Validation v{iv.__version__} installed successfully!")

# Check available validators
print(f"Available validators: {len(iv.VALIDATION_MODULES)}")
for module in iv.VALIDATION_MODULES:
    print(f"  - {module}")
```

## Quick Start

### 1. Run Complete Validation Suite

```python
import imhotep_validation as iv

# Run comprehensive validation of all claims
results = iv.validate_all_claims(
    output_dir="./validation_results",
    parallel_validation=True,
    verbose=True
)

# Check overall results
summary = results['comprehensive_summary']
print(f"Overall validation: {'PASSED' if summary['overall_validation_passed'] else 'FAILED'}")
print(f"Frameworks validated: {summary['framework_statistics']['frameworks_passed']}/{summary['framework_statistics']['total_frameworks']}")
```

### 2. Run Quick Validation

```python
# Run quick validation for fast feedback
quick_results = iv.quick_validation()
print(f"Quick validation: {'PASSED' if quick_results['validation_passed'] else 'FAILED'}")
```

### 3. Run Individual Validators

```python
from imhotep_validation.validators import UniversalProblemSolvingValidator

# Create and run specific validator
validator = UniversalProblemSolvingValidator(verbose=True, random_seed=42)
results = validator.run_validation(save_results=True)

print(f"Universal Problem-Solving validation: {'PASSED' if results['validation_results']['validation_passed'] else 'FAILED'}")
```

## Command Line Interface

### List Available Validators

```bash
python -m imhotep_validation.main list
```

### Run Complete Validation Suite

```bash
python -m imhotep_validation.main run-all --output ./results --verbose
```

### Run Specific Validator

```bash
python -m imhotep_validation.main run universal_problem_solving --verbose
python -m imhotep_validation.main run bmd_information_catalysis --output ./bmd_results
```

### Command Line Options

- `--output DIR`: Specify output directory for results
- `--verbose`: Enable detailed progress output
- `--quiet`: Suppress most output
- `--sequential`: Run validations sequentially instead of in parallel
- `--seed N`: Set random seed for reproducible results

## Understanding Results

### Validation Result Structure

Each validation returns a comprehensive result dictionary:

```python
{
    'metadata': {
        'validator_name': 'example_validator',
        'start_time': '2024-01-01T12:00:00Z',
        'duration_seconds': 45.2,
        # ... more metadata
    },
    'validation_results': {
        'validation_passed': True,
        'confidence': 0.95,
        'p_value': 0.0001,
        'effect_size': 1.2,
        'claims_tested': 4,
        'claims_validated': 4,
        'detailed_results': {
            # Detailed results for each claim tested
        }
    },
    'summary': {
        'status': 'PASSED',
        'validation_rate': 1.0,
        # ... summary statistics
    }
}
```

### Key Metrics

- **validation_passed**: Boolean indicating if all claims passed validation
- **confidence**: Overall confidence level (0.0 to 1.0)
- **p_value**: Statistical significance (smaller is better, typically < 0.001)
- **effect_size**: Magnitude of effects observed (typically > 0.8 for large effects)
- **claims_validated/claims_tested**: Number of claims that passed validation

### Interpreting Results

- **PASSED**: All claims validated with high confidence (p < 0.001, large effect sizes)
- **FAILED**: One or more claims failed validation or showed insufficient evidence
- **ERROR**: Technical error occurred during validation

## Validation Modules

### Universal Problem-Solving Engine
Tests the theoretical claims about reality as a universal problem-solving engine:
- Computational impossibility of real-time reality generation
- Universal solvability theorem
- Navigation vs computation indeterminability

### BMD Information Catalysis  
Tests Biological Maxwell's Demons and information catalysis claims:
- Information processing speedup
- Frame selection efficiency
- Catalytic information processing

### Quantum Membrane Dynamics
Tests quantum membrane dynamics and ENAQT claims:
- Room-temperature quantum coherence
- ENAQT ion channel efficiency  
- Quantum information processing advantages

### Self-Awareness
Tests genuine self-awareness and metacognitive monitoring:
- Four-file system architecture
- Metacognitive monitoring accuracy
- Uncertainty quantification
- Genuine vs simulated self-awareness

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'imhotep_validation'
   ```
   Solution: Ensure the package is installed (`pip install -e .` in demo directory)

2. **Missing Dependencies**
   ```
   ModuleNotFoundError: No module named 'numpy'
   ```
   Solution: Install dependencies (`pip install -r requirements.txt`)

3. **Permission Errors**
   ```
   PermissionError: [Errno 13] Permission denied: './validation_results'
   ```
   Solution: Use a different output directory or check permissions

4. **Memory Issues**
   ```
   MemoryError: Unable to allocate array
   ```
   Solution: Use `parallel_validation=False` or reduce the number of trials

### Getting Help

If you encounter issues:

1. Check that all dependencies are installed: `pip list`
2. Verify Python version: `python --version` (must be ≥ 3.7)
3. Check package status:
   ```python
   import imhotep_validation as iv
   print(iv.get_validation_status())
   ```

### Performance Notes

- **Parallel Validation**: Faster but uses more memory
- **Sequential Validation**: Slower but more memory-efficient
- **Random Seeds**: Use consistent seeds for reproducible results
- **Output Directory**: Results are saved in JSON format for later analysis

## Advanced Usage

### Custom Validation Parameters

```python
from imhotep_validation.core import ComprehensiveValidator

validator = ComprehensiveValidator(
    output_dir="./custom_results",
    parallel_validation=True,
    verbose=True,
    random_seed=12345  # Custom seed for reproducibility
)

results = validator.validate_all_claims(
    save_results=True,
    detailed_analysis=True,
    generate_report=True
)
```

### Accessing Individual Validators

```python
from imhotep_validation.validators import ALL_VALIDATORS

# List all available validators
for validator_class in ALL_VALIDATORS:
    validator = validator_class(verbose=False)
    print(f"Validator: {validator.name}")
    print(f"Description: {validator.description}")
```

### Statistical Analysis

```python
from imhotep_validation.utils import calculate_entropy
import numpy as np

# Use utility functions for custom analysis
data = np.random.rand(1000)
entropy = calculate_entropy(data)
print(f"Data entropy: {entropy:.3f}")
```

This completes the installation and usage guide for the Imhotep Validation Package.
