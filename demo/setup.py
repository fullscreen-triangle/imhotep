#!/usr/bin/env python3
"""
Imhotep Framework Validation Package
====================================

Comprehensive experimental validation of all theoretical claims in the Imhotep neural architecture framework.
This package provides rigorous experimental validation for:

1. Universal Problem-Solving Engine Theory
2. BMD Information Catalysis Framework
3. Quantum Membrane Dynamics
4. Visual Consciousness Framework
5. Pharmaceutical Consciousness Optimization
6. Self-Aware Neural Networks
7. Production Performance Claims
8. All quantitative performance metrics

Author: Kundai Farai Sachikonye
Institution: Independent Research Institute, Buhera, Zimbabwe
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Comprehensive experimental validation package for Imhotep neural architecture framework"

setup(
    name="imhotep-validation",
    version="1.0.0",
    author="Kundai Farai Sachikonye",
    author_email="kundai.sachikonye@wzw.tum.de",
    description="Comprehensive experimental validation of Imhotep neural architecture theoretical claims",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/kundai-sachikonye/imhotep-validation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Biology",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "torch>=1.10.0",
        "tensorflow>=2.6.0",
        "qiskit>=0.34.0",
        "networkx>=2.6.0",
        "plotly>=5.0.0",
        "dash>=2.0.0",
        "jupyter>=1.0.0",
        "ipython>=7.0.0",
        "tqdm>=4.60.0",
        "pillow>=8.0.0",
        "opencv-python>=4.5.0",
        "librosa>=0.8.0",
        "rdkit>=2022.03.0",
        "biopython>=1.78",
        "sympy>=1.8.0",
        "statsmodels>=0.12.0",
        "pytest>=6.0.0",
        "pytest-cov>=2.12.0",
        "black>=21.0.0",
        "flake8>=3.9.0",
        "mypy>=0.910",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
        "gpu": [
            "cupy>=9.0.0",
            "cudf>=21.06.0",
        ],
        "quantum": [
            "cirq>=0.12.0",
            "pennylane>=0.20.0",
            "qiskit-aer>=0.9.0",
            "qiskit-optimization>=0.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "imhotep-validate=imhotep_validation.cli:main",
            "imhotep-benchmark=imhotep_validation.benchmarking:main",
            "imhotep-visualize=imhotep_validation.visualization:main",
        ],
    },
    include_package_data=True,
    package_data={
        "imhotep_validation": [
            "data/*.json",
            "data/*.csv",
            "data/*.npy",
            "templates/*.html",
            "configs/*.yaml",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/kundai-sachikonye/imhotep-validation/issues",
        "Source": "https://github.com/kundai-sachikonye/imhotep-validation",
        "Documentation": "https://imhotep-validation.readthedocs.io/",
        "Research Paper": "https://arxiv.org/abs/2024.imhotep",
    },
)
