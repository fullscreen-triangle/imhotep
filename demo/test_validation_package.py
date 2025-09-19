#!/usr/bin/env python3
"""
Test script for the Imhotep validation package.

This script tests the complete validation package to ensure all components
work correctly together.
"""

import sys
import os
import traceback

# Add the demo directory to the Python path so we can import the package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported correctly."""
    print("🧪 Testing imports...")

    try:
        # Test core imports
        from imhotep_validation.core import ValidationBase, ComprehensiveValidator, QuickValidator
        print("✅ Core modules imported successfully")

        # Test validator imports
        from imhotep_validation.validators import (
            UniversalProblemSolvingValidator,
            BMDInformationCatalysisValidator,
            QuantumMembraneDynamicsValidator,
            SelfAwarenessValidator,
            ALL_VALIDATORS
        )
        print(f"✅ Individual validators imported successfully ({len(ALL_VALIDATORS)} validators)")

        # Test utility imports
        from imhotep_validation.utils import calculate_entropy
        print("✅ Utility functions imported successfully")

        # Test main package import
        import imhotep_validation as iv
        print("✅ Main package imported successfully")

        return True

    except Exception as e:
        print(f"❌ Import test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_individual_validators():
    """Test that individual validators can be created and run."""
    print("\n🧪 Testing individual validators...")

    try:
        from imhotep_validation.validators import ALL_VALIDATORS

        success_count = 0
        for validator_class in ALL_VALIDATORS:
            try:
                # Create validator instance
                validator = validator_class(verbose=False, random_seed=42)
                print(f"📋 Testing {validator.name}...")

                # Run a quick validation
                results = validator.validate()

                if results.get('validation_passed', False):
                    print(f"✅ {validator.name}: PASSED")
                else:
                    print(f"⚠️ {validator.name}: FAILED (but validator executed correctly)")

                success_count += 1

            except Exception as e:
                print(f"❌ {validator_class.__name__} failed: {str(e)}")

        print(f"📊 Individual validator test: {success_count}/{len(ALL_VALIDATORS)} validators executed successfully")
        return success_count == len(ALL_VALIDATORS)

    except Exception as e:
        print(f"❌ Individual validator test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_comprehensive_validation():
    """Test the comprehensive validation system."""
    print("\n🧪 Testing comprehensive validation...")

    try:
        from imhotep_validation.core import ComprehensiveValidator

        # Create comprehensive validator
        validator = ComprehensiveValidator(
            output_dir="./test_results",
            verbose=False,
            random_seed=42,
            parallel_validation=False  # Use sequential for testing
        )

        print("📋 Running comprehensive validation...")

        # Run comprehensive validation
        results = validator.validate_all_claims(
            save_results=False,  # Don't save during testing
            detailed_analysis=True,
            generate_report=False  # Don't generate report during testing
        )

        # Check results
        if results:
            summary = results.get('comprehensive_summary', {})
            frameworks_passed = summary.get('framework_statistics', {}).get('frameworks_passed', 0)
            total_frameworks = summary.get('framework_statistics', {}).get('total_frameworks', 0)

            print(f"📊 Comprehensive validation: {frameworks_passed}/{total_frameworks} frameworks executed")
            print(f"✅ Comprehensive validation system works correctly")
            return True
        else:
            print("❌ Comprehensive validation returned no results")
            return False

    except Exception as e:
        print(f"❌ Comprehensive validation test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_quick_validation():
    """Test the quick validation system."""
    print("\n🧪 Testing quick validation...")

    try:
        from imhotep_validation.core import QuickValidator

        # Create quick validator
        validator = QuickValidator(verbose=False, random_seed=42)

        print("📋 Running quick validation...")

        # Run quick validation
        results = validator.run_validation(save_results=False)

        # Check results
        if results:
            validation_results = results.get('validation_results', {})
            claims_validated = validation_results.get('claims_validated', 0)
            claims_tested = validation_results.get('claims_tested', 0)

            print(f"📊 Quick validation: {claims_validated}/{claims_tested} claims validated")
            print(f"✅ Quick validation system works correctly")
            return True
        else:
            print("❌ Quick validation returned no results")
            return False

    except Exception as e:
        print(f"❌ Quick validation test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_main_entry_point():
    """Test the main entry point functionality."""
    print("\n🧪 Testing main entry point...")

    try:
        from imhotep_validation.main import list_validators

        # Test listing validators
        print("📋 Testing validator listing...")
        list_validators()
        print("✅ Main entry point works correctly")
        return True

    except Exception as e:
        print(f"❌ Main entry point test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_package_interface():
    """Test the main package interface functions."""
    print("\n🧪 Testing package interface...")

    try:
        import imhotep_validation as iv

        # Test main functions exist
        assert hasattr(iv, 'validate_all_claims'), "validate_all_claims function not found"
        assert hasattr(iv, 'quick_validation'), "quick_validation function not found"
        assert hasattr(iv, 'get_validation_status'), "get_validation_status function not found"

        # Test status function
        status = iv.get_validation_status()
        print(f"📊 Validation environment status: {status}")

        print("✅ Package interface works correctly")
        return True

    except Exception as e:
        print(f"❌ Package interface test failed: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧠 Starting Imhotep Validation Package Test Suite")
    print("=" * 60)

    tests = [
        ("Import Test", test_imports),
        ("Individual Validators Test", test_individual_validators),
        ("Comprehensive Validation Test", test_comprehensive_validation),
        ("Quick Validation Test", test_quick_validation),
        ("Main Entry Point Test", test_main_entry_point),
        ("Package Interface Test", test_package_interface)
    ]

    passed_tests = 0
    total_tests = len(tests)

    for test_name, test_function in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            if test_function():
                passed_tests += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} CRASHED: {str(e)}")

    # Final summary
    print("\n" + "=" * 60)
    print("🧠 IMHOTEP VALIDATION PACKAGE TEST RESULTS")
    print("=" * 60)
    print(f"Tests passed: {passed_tests}/{total_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")

    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! The validation package is ready for use.")
        return True
    else:
        print("⚠️ Some tests failed. Please review the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
