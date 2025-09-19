#!/usr/bin/env python3
"""
Basic usage examples for the Imhotep Validation Package.

This script demonstrates how to use the validation package to test
theoretical claims in the Imhotep framework.
"""

import sys
import os

# Add demo directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def example_comprehensive_validation():
    """Example: Run comprehensive validation of all claims."""
    print("🧠 Example 1: Comprehensive Validation")
    print("=" * 50)

    try:
        import imhotep_validation as iv

        # Run comprehensive validation
        print("Running comprehensive validation...")
        results = iv.validate_all_claims(
            output_dir="./example_results",
            parallel_validation=True,
            verbose=True,
            random_seed=42
        )

        # Extract and display key results
        summary = results['comprehensive_summary']
        print(f"\n📊 RESULTS:")
        print(f"Overall Status: {'✅ PASSED' if summary['overall_validation_passed'] else '❌ FAILED'}")

        framework_stats = summary['framework_statistics']
        print(f"Frameworks: {framework_stats['frameworks_passed']}/{framework_stats['total_frameworks']} passed")

        claims_stats = summary['claims_statistics']
        print(f"Claims: {claims_stats['total_claims_validated']}/{claims_stats['total_claims_tested']} validated")

        metrics = summary['statistical_metrics']
        print(f"Confidence: {metrics['overall_confidence']:.1%}")
        print(f"P-value: {metrics['combined_p_value']:.2e}")
        print(f"Effect size: {metrics['overall_effect_size']:.2f}")

        return True

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def example_individual_validators():
    """Example: Run individual validators."""
    print("\n🧠 Example 2: Individual Validators")
    print("=" * 50)

    try:
        from imhotep_validation.validators import (
            UniversalProblemSolvingValidator,
            BMDInformationCatalysisValidator,
            QuantumMembraneDynamicsValidator,
            SelfAwarenessValidator
        )

        # Test each validator individually
        validators = [
            UniversalProblemSolvingValidator,
            BMDInformationCatalysisValidator,
            QuantumMembraneDynamicsValidator,
            SelfAwarenessValidator
        ]

        results = {}
        for validator_class in validators:
            print(f"\n📋 Testing {validator_class.__name__}...")

            validator = validator_class(verbose=False, random_seed=42)
            result = validator.validate()

            results[validator.name] = result
            status = "✅ PASSED" if result['validation_passed'] else "❌ FAILED"
            confidence = result.get('confidence', 0)

            print(f"   {status} (Confidence: {confidence:.1%})")

        print(f"\n📊 Individual Validation Summary:")
        passed = sum(1 for r in results.values() if r['validation_passed'])
        total = len(results)
        print(f"   {passed}/{total} validators passed")

        return True

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def example_quick_validation():
    """Example: Run quick validation."""
    print("\n🧠 Example 3: Quick Validation")
    print("=" * 50)

    try:
        import imhotep_validation as iv

        print("Running quick validation...")
        results = iv.quick_validation(verbose=False, random_seed=42)

        # Display results
        validation_results = results['validation_results']
        status = "✅ PASSED" if validation_results['validation_passed'] else "❌ FAILED"
        confidence = validation_results.get('confidence', 0)
        claims_validated = validation_results.get('claims_validated', 0)
        claims_tested = validation_results.get('claims_tested', 0)

        print(f"\n📊 Quick Validation Results:")
        print(f"   Status: {status}")
        print(f"   Confidence: {confidence:.1%}")
        print(f"   Claims: {claims_validated}/{claims_tested} validated")

        return True

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def example_custom_validation():
    """Example: Custom validation with specific parameters."""
    print("\n🧠 Example 4: Custom Validation")
    print("=" * 50)

    try:
        from imhotep_validation.core import ComprehensiveValidator

        # Create custom validator with specific parameters
        validator = ComprehensiveValidator(
            output_dir="./custom_validation_results",
            parallel_validation=False,  # Sequential for this example
            verbose=True,
            random_seed=12345  # Custom seed
        )

        print("Running custom validation...")
        results = validator.validate_all_claims(
            save_results=False,  # Don't save files in this example
            detailed_analysis=True,
            generate_report=False
        )

        # Access detailed framework results
        framework_results = results.get('framework_results', {})
        print(f"\n📊 Detailed Framework Results:")

        for framework_name, framework_result in framework_results.items():
            validation_result = framework_result['validation_results']
            status = "✅" if validation_result.get('validation_passed', False) else "❌"
            confidence = validation_result.get('confidence', 0)

            print(f"   {status} {framework_name}: {confidence:.1%} confidence")

        return True

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def example_statistical_analysis():
    """Example: Using statistical utilities."""
    print("\n🧠 Example 5: Statistical Analysis")
    print("=" * 50)

    try:
        from imhotep_validation.utils import calculate_entropy
        import numpy as np

        # Generate sample data
        np.random.seed(42)
        uniform_data = np.random.uniform(0, 1, 1000)
        gaussian_data = np.random.normal(0.5, 0.1, 1000)

        # Calculate entropy
        uniform_entropy = calculate_entropy(uniform_data)
        gaussian_entropy = calculate_entropy(gaussian_data)

        print(f"📊 Entropy Analysis:")
        print(f"   Uniform data entropy: {uniform_entropy:.3f}")
        print(f"   Gaussian data entropy: {gaussian_entropy:.3f}")
        print(f"   Entropy difference: {abs(uniform_entropy - gaussian_entropy):.3f}")

        return True

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def main():
    """Run all examples."""
    print("🧠 IMHOTEP VALIDATION PACKAGE - USAGE EXAMPLES")
    print("=" * 60)

    examples = [
        ("Comprehensive Validation", example_comprehensive_validation),
        ("Individual Validators", example_individual_validators),
        ("Quick Validation", example_quick_validation),
        ("Custom Validation", example_custom_validation),
        ("Statistical Analysis", example_statistical_analysis)
    ]

    success_count = 0
    for name, example_func in examples:
        print(f"\n{'='*60}")
        try:
            if example_func():
                success_count += 1
                print(f"✅ {name} example completed successfully")
            else:
                print(f"❌ {name} example failed")
        except Exception as e:
            print(f"💥 {name} example crashed: {str(e)}")

    # Final summary
    print(f"\n{'='*60}")
    print("📊 EXAMPLES SUMMARY")
    print(f"Examples completed successfully: {success_count}/{len(examples)}")

    if success_count == len(examples):
        print("🎉 All examples ran successfully!")
    else:
        print("⚠️ Some examples had issues. Check the output above for details.")

if __name__ == "__main__":
    main()
