#!/usr/bin/env python3
"""
Main entry point for Imhotep validation package.

This module provides the command-line interface for running comprehensive
validation of all Imhotep framework theoretical claims.

Usage:
    python -m imhotep_validation.main run-all
    python -m imhotep_validation.main run <validator_name>
    python -m imhotep_validation.main list
    python -m imhotep_validation.main --help
"""

import sys
import argparse
import logging
from typing import Optional, List
from pathlib import Path

from .core.comprehensive_validator import ComprehensiveValidator
from .validators import ALL_VALIDATORS


def setup_logging(verbose: bool = True):
    """Setup logging for the validation suite."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def list_validators() -> None:
    """List all available validators."""
    print("\n🧠 Available Imhotep Validators:")
    print("=" * 50)
    
    for i, validator_class in enumerate(ALL_VALIDATORS, 1):
        # Create a temporary instance to get description
        validator = validator_class(verbose=False)
        print(f"{i}. {validator.name}")
        print(f"   Description: {validator.description}")
        print()


def run_single_validator(validator_name: str, output_dir: Optional[str] = None, 
                        verbose: bool = True, random_seed: int = 42) -> None:
    """Run a single validator by name."""
    # Find the validator class
    validator_class = None
    for cls in ALL_VALIDATORS:
        temp_validator = cls(verbose=False)
        if temp_validator.name.lower().replace(' ', '_') == validator_name.lower().replace('-', '_'):
            validator_class = cls
            break
    
    if validator_class is None:
        print(f"❌ Validator '{validator_name}' not found.")
        print("\nAvailable validators:")
        list_validators()
        return
    
    print(f"\n🧪 Running single validation: {validator_name}")
    print("=" * 60)
    
    try:
        # Create validator instance with proper parameters
        validator = validator_class(
            verbose=verbose, 
            random_seed=random_seed, 
            output_dir=output_dir
        )
        
        # Run validation
        results = validator.run_validation(save_results=True)
        
        # Display results
        print(f"\n✅ Validation completed for: {validator.name}")
        summary = results.get('summary', {})
        print(f"Status: {'PASSED' if summary.get('status') == 'PASSED' else 'FAILED'}")
        print(f"Confidence: {summary.get('confidence', 0):.3f}")
        print(f"Claims validated: {summary.get('claims_validated', 0)}/{summary.get('claims_tested', 0)}")
        print(f"Duration: {summary.get('duration', 0):.2f} seconds")
        
        # Show detailed results if available
        validation_results = results.get('validation_results', {})
        if 'detailed_results' in validation_results:
            print(f"\nDetailed Results Summary:")
            detailed = validation_results['detailed_results']
            for key, value in detailed.items():
                if isinstance(value, dict) and 'summary' in value:
                    print(f"- {key}: {value['summary']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Failed to run validator '{validator_name}': {str(e)}")
        return None


def run_comprehensive_validation(output_dir: Optional[str] = None, 
                                verbose: bool = True, 
                                parallel: bool = True,
                                random_seed: int = 42) -> None:
    """Run comprehensive validation of all claims."""
    print("\n🧠 Starting Comprehensive Imhotep Framework Validation")
    print("=" * 80)
    print(f"🔬 Testing {len(ALL_VALIDATORS)} theoretical frameworks")
    print(f"⚡ Parallel execution: {'Enabled' if parallel else 'Disabled'}")
    print(f"🎲 Random seed: {random_seed}")
    print("=" * 80)
    
    # Create comprehensive validator
    validator = ComprehensiveValidator(
        output_dir=output_dir,
        parallel_validation=parallel,
        verbose=verbose,
        random_seed=random_seed
    )
    
    try:
        # Run comprehensive validation
        results = validator.validate_all_claims(
            save_results=True,
            detailed_analysis=True,
            generate_report=True
        )
        
        # Display summary
        print(f"\n🎉 Comprehensive validation completed!")
        
        summary = results.get('comprehensive_summary', {})
        if summary:
            overall_passed = summary.get('overall_validation_passed', False)
            print(f"Overall Status: {'✅ ALL CLAIMS VALIDATED' if overall_passed else '❌ VALIDATION ISSUES DETECTED'}")
            
            framework_stats = summary.get('framework_statistics', {})
            print(f"Frameworks: {framework_stats.get('frameworks_passed', 0)}/{framework_stats.get('total_frameworks', 0)} passed")
            
            claims_stats = summary.get('claims_statistics', {})
            print(f"Claims: {claims_stats.get('total_claims_validated', 0)}/{claims_stats.get('total_claims_tested', 0)} validated")
            
            stat_metrics = summary.get('statistical_metrics', {})
            print(f"Confidence: {stat_metrics.get('overall_confidence', 0):.3f}")
            print(f"P-value: {stat_metrics.get('combined_p_value', 1):.6f}")
            print(f"Effect size: {stat_metrics.get('overall_effect_size', 0):.3f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Comprehensive validation failed: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None


def main():
    """Main entry point for the validation suite."""
    parser = argparse.ArgumentParser(
        description="Imhotep Framework Validation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s run-all                    # Run comprehensive validation
  %(prog)s run universal_problem_solving  # Run specific validator
  %(prog)s list                       # List all available validators
  %(prog)s run-all --output results   # Save results to 'results' directory
        """
    )
    
    parser.add_argument(
        'command', 
        choices=['run-all', 'run', 'list'],
        help='Command to execute'
    )
    
    parser.add_argument(
        'validator_name', 
        nargs='?',
        help='Name of specific validator to run (required for "run" command)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory for results (default: ./validation_results)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress most output'
    )
    
    parser.add_argument(
        '--sequential',
        action='store_true',
        help='Run validations sequentially instead of in parallel'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible results (default: 42)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup verbosity
    verbose = args.verbose and not args.quiet
    setup_logging(verbose)
    
    # Handle commands
    if args.command == 'list':
        list_validators()
        
    elif args.command == 'run':
        if not args.validator_name:
            print("❌ Error: validator name is required for 'run' command")
            parser.print_help()
            sys.exit(1)
            
        run_single_validator(
            args.validator_name,
            output_dir=args.output,
            verbose=verbose,
            random_seed=args.seed
        )
        
    elif args.command == 'run-all':
        run_comprehensive_validation(
            output_dir=args.output,
            verbose=verbose,
            parallel=not args.sequential,
            random_seed=args.seed
        )
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
