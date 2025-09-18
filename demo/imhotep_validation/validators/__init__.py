"""
Individual validation modules for each theoretical framework in the Imhotep system.

Each validator implements rigorous experimental validation of specific theoretical claims:
- Universal Problem-Solving Engine Theory
- BMD Information Catalysis Framework
- Quantum Membrane Dynamics
- Visual Consciousness Framework
- Pharmaceutical Consciousness Optimization
- Self-Aware Neural Networks
- Production Performance Claims
"""

from .universal_problem_solving import UniversalProblemSolvingValidator
from .bmd_information_catalysis import BMDInformationCatalysisValidator
from .quantum_membrane_dynamics import QuantumMembraneDynamicsValidator
from .self_awareness import SelfAwarenessValidator
# from .visual_consciousness import VisualConsciousnessValidator
# from .pharmaceutical_optimization import PharmaceuticalOptimizationValidator
# from .production_performance import ProductionPerformanceValidator

# List of all available validators
ALL_VALIDATORS = [
    UniversalProblemSolvingValidator,
    BMDInformationCatalysisValidator,
    QuantumMembraneDynamicsValidator,
    SelfAwarenessValidator,
    # VisualConsciousnessValidator,
    # PharmaceuticalOptimizationValidator,
    # ProductionPerformanceValidator
]

__all__ = [
    'UniversalProblemSolvingValidator',
    'BMDInformationCatalysisValidator',
    'QuantumMembraneDynamicsValidator',
    'SelfAwarenessValidator',
    'ALL_VALIDATORS'
    # 'VisualConsciousnessValidator',
    # 'PharmaceuticalOptimizationValidator',
    # 'ProductionPerformanceValidator'
]
