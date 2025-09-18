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
from .visual_consciousness import VisualConsciousnessValidator
from .pharmaceutical_optimization import PharmaceuticalOptimizationValidator
from .self_aware_networks import SelfAwareNetworksValidator
from .production_performance import ProductionPerformanceValidator

__all__ = [
    'UniversalProblemSolvingValidator',
    'BMDInformationCatalysisValidator',
    'QuantumMembraneDynamicsValidator',
    'VisualConsciousnessValidator',
    'PharmaceuticalOptimizationValidator',
    'SelfAwareNetworksValidator',
    'ProductionPerformanceValidator'
]
