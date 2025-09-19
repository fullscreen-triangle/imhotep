#!/usr/bin/env python3
"""
Statistical Analysis Utilities for Imhotep Validation

This module provides comprehensive statistical analysis capabilities for validating
theoretical claims in the Imhotep framework, including significance testing,
effect size calculation, power analysis, and meta-analysis methods.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Any, Tuple, Optional, Union
import warnings
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class StatisticalResult:
    """Container for statistical analysis results."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    interpretation: str
    sample_size: int
    power: Optional[float] = None


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis utilities for Imhotep validation.

    Provides methods for:
    - Significance testing (t-tests, ANOVA, non-parametric tests)
    - Effect size calculation (Cohen's d, eta-squared, etc.)
    - Power analysis and sample size calculation
    - Meta-analysis for combining results
    - Bayesian analysis methods
    - Multiple comparisons correction
    """

    def __init__(self, alpha: float = 0.001, confidence_level: float = 0.95):
        """
        Initialize statistical analyzer.

        Args:
            alpha: Significance threshold (default: 0.001 for high confidence)
            confidence_level: Confidence level for intervals (default: 0.95)
        """
        self.alpha = alpha
        self.confidence_level = confidence_level

    def compare_groups(self,
                      experimental_data: np.ndarray,
                      control_data: np.ndarray,
                      test_type: str = 'auto') -> StatisticalResult:
        """
        Compare two groups with comprehensive statistical analysis.

        Args:
            experimental_data: Data from experimental condition
            control_data: Data from control condition
            test_type: Type of test ('auto', 't_test', 'mann_whitney', 'welch')

        Returns:
            Complete statistical analysis results
        """
        # Determine appropriate test
        if test_type == 'auto':
            test_type = self._select_appropriate_test(experimental_data, control_data)

        # Perform the test
        if test_type == 't_test':
            result = self._independent_t_test(experimental_data, control_data)
        elif test_type == 'welch':
            result = self._welch_t_test(experimental_data, control_data)
        elif test_type == 'mann_whitney':
            result = self._mann_whitney_test(experimental_data, control_data)
        else:
            raise ValueError(f"Unknown test type: {test_type}")

        return result

    def _select_appropriate_test(self,
                                experimental_data: np.ndarray,
                                control_data: np.ndarray) -> str:
        """Automatically select appropriate statistical test."""
        # Check normality
        exp_normal = self._test_normality(experimental_data)
        ctrl_normal = self._test_normality(control_data)

        # Check equal variances
        equal_variances = self._test_equal_variances(experimental_data, control_data)

        if exp_normal and ctrl_normal:
            if equal_variances:
                return 't_test'
            else:
                return 'welch'
        else:
            return 'mann_whitney'

    def _test_normality(self, data: np.ndarray) -> bool:
        """Test if data follows normal distribution."""
        if len(data) < 8:
            return True  # Assume normal for small samples

        # Shapiro-Wilk test for normality
        _, p_value = stats.shapiro(data)
        return p_value > 0.05  # Not significantly different from normal

    def _test_equal_variances(self,
                             data1: np.ndarray,
                             data2: np.ndarray) -> bool:
        """Test if two groups have equal variances."""
        # Levene's test for equal variances
        _, p_value = stats.levene(data1, data2)
        return p_value > 0.05  # Not significantly different variances

    def _independent_t_test(self,
                           experimental_data: np.ndarray,
                           control_data: np.ndarray) -> StatisticalResult:
        """Perform independent samples t-test."""
        # Calculate descriptive statistics
        exp_mean = np.mean(experimental_data)
        ctrl_mean = np.mean(control_data)
        exp_std = np.std(experimental_data, ddof=1)
        ctrl_std = np.std(control_data, ddof=1)

        n1, n2 = len(experimental_data), len(control_data)

        # Perform t-test
        t_stat, p_value = stats.ttest_ind(experimental_data, control_data)

        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((n1 - 1) * exp_std**2 + (n2 - 1) * ctrl_std**2) / (n1 + n2 - 2))
        cohens_d = (exp_mean - ctrl_mean) / pooled_std

        # Confidence interval for the difference
        se_diff = pooled_std * np.sqrt(1/n1 + 1/n2)
        df = n1 + n2 - 2
        t_critical = stats.t.ppf((1 + self.confidence_level) / 2, df)

        mean_diff = exp_mean - ctrl_mean
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff

        # Interpret effect size
        interpretation = self._interpret_cohens_d(cohens_d)

        # Calculate power
        power = self._calculate_power_t_test(cohens_d, n1, n2, self.alpha)

        return StatisticalResult(
            test_name='Independent t-test',
            statistic=t_stat,
            p_value=p_value,
            effect_size=abs(cohens_d),
            confidence_interval=(ci_lower, ci_upper),
            is_significant=p_value < self.alpha,
            interpretation=interpretation,
            sample_size=n1 + n2,
            power=power
        )

    def _welch_t_test(self,
                     experimental_data: np.ndarray,
                     control_data: np.ndarray) -> StatisticalResult:
        """Perform Welch's t-test (unequal variances)."""
        # Calculate descriptive statistics
        exp_mean = np.mean(experimental_data)
        ctrl_mean = np.mean(control_data)
        exp_var = np.var(experimental_data, ddof=1)
        ctrl_var = np.var(control_data, ddof=1)

        n1, n2 = len(experimental_data), len(control_data)

        # Perform Welch's t-test
        t_stat, p_value = stats.ttest_ind(experimental_data, control_data, equal_var=False)

        # Calculate effect size (Cohen's d with pooled standard deviation)
        exp_std = np.sqrt(exp_var)
        ctrl_std = np.sqrt(ctrl_var)
        pooled_std = np.sqrt((exp_var + ctrl_var) / 2)
        cohens_d = (exp_mean - ctrl_mean) / pooled_std

        # Welch-Satterthwaite degrees of freedom
        df = (exp_var/n1 + ctrl_var/n2)**2 / ((exp_var/n1)**2/(n1-1) + (ctrl_var/n2)**2/(n2-1))

        # Confidence interval
        se_diff = np.sqrt(exp_var/n1 + ctrl_var/n2)
        t_critical = stats.t.ppf((1 + self.confidence_level) / 2, df)

        mean_diff = exp_mean - ctrl_mean
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff

        interpretation = self._interpret_cohens_d(cohens_d)
        power = self._calculate_power_t_test(cohens_d, n1, n2, self.alpha)

        return StatisticalResult(
            test_name="Welch's t-test",
            statistic=t_stat,
            p_value=p_value,
            effect_size=abs(cohens_d),
            confidence_interval=(ci_lower, ci_upper),
            is_significant=p_value < self.alpha,
            interpretation=interpretation,
            sample_size=n1 + n2,
            power=power
        )

    def _mann_whitney_test(self,
                          experimental_data: np.ndarray,
                          control_data: np.ndarray) -> StatisticalResult:
        """Perform Mann-Whitney U test (non-parametric)."""
        # Perform Mann-Whitney U test
        u_stat, p_value = stats.mannwhitneyu(experimental_data, control_data, alternative='two-sided')

        n1, n2 = len(experimental_data), len(control_data)

        # Calculate effect size (rank-biserial correlation)
        # r = 1 - (2*U)/(n1*n2)
        effect_size = 1 - (2 * u_stat) / (n1 * n2)

        # For confidence interval, we'll use the median difference
        combined_data = np.concatenate([experimental_data, control_data])
        group_labels = np.concatenate([np.ones(n1), np.zeros(n2)])

        # Bootstrap confidence interval for median difference
        ci_lower, ci_upper = self._bootstrap_median_difference_ci(
            experimental_data, control_data, self.confidence_level
        )

        # Interpret effect size
        interpretation = self._interpret_rank_biserial_correlation(effect_size)

        return StatisticalResult(
            test_name='Mann-Whitney U test',
            statistic=u_stat,
            p_value=p_value,
            effect_size=abs(effect_size),
            confidence_interval=(ci_lower, ci_upper),
            is_significant=p_value < self.alpha,
            interpretation=interpretation,
            sample_size=n1 + n2,
            power=None  # Power calculation complex for non-parametric tests
        )

    def _bootstrap_median_difference_ci(self,
                                       data1: np.ndarray,
                                       data2: np.ndarray,
                                       confidence_level: float) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for median difference."""
        n_bootstrap = 1000
        median_diffs = []

        for _ in range(n_bootstrap):
            # Bootstrap samples
            bootstrap_data1 = np.random.choice(data1, size=len(data1), replace=True)
            bootstrap_data2 = np.random.choice(data2, size=len(data2), replace=True)

            # Calculate median difference
            median_diff = np.median(bootstrap_data1) - np.median(bootstrap_data2)
            median_diffs.append(median_diff)

        # Calculate confidence interval
        alpha_level = 1 - confidence_level
        ci_lower = np.percentile(median_diffs, 100 * alpha_level / 2)
        ci_upper = np.percentile(median_diffs, 100 * (1 - alpha_level / 2))

        return ci_lower, ci_upper

    def _interpret_cohens_d(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def _interpret_rank_biserial_correlation(self, r: float) -> str:
        """Interpret rank-biserial correlation effect size."""
        abs_r = abs(r)
        if abs_r < 0.1:
            return "negligible"
        elif abs_r < 0.3:
            return "small"
        elif abs_r < 0.5:
            return "medium"
        else:
            return "large"

    def _calculate_power_t_test(self,
                               effect_size: float,
                               n1: int,
                               n2: int,
                               alpha: float) -> float:
        """Calculate statistical power for t-test."""
        # Simplified power calculation using normal approximation
        # More precise calculation would require statsmodels or specialized libraries

        pooled_n = (n1 * n2) / (n1 + n2)
        ncp = effect_size * np.sqrt(pooled_n / 2)  # Non-centrality parameter

        # Critical t-value
        df = n1 + n2 - 2
        t_critical = stats.t.ppf(1 - alpha / 2, df)

        # Power calculation using non-central t-distribution
        # Approximation: power ≈ 1 - Φ(t_critical - ncp) + Φ(-t_critical - ncp)
        power = 1 - stats.norm.cdf(t_critical - ncp) + stats.norm.cdf(-t_critical - ncp)

        return min(power, 1.0)  # Cap at 1.0

    def one_way_anova(self, *groups) -> StatisticalResult:
        """Perform one-way ANOVA for multiple groups."""
        # Perform ANOVA
        f_stat, p_value = stats.f_oneway(*groups)

        # Calculate effect size (eta-squared)
        grand_mean = np.mean(np.concatenate(groups))

        # Between-group sum of squares
        ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in groups)

        # Total sum of squares
        ss_total = sum(np.sum((group - grand_mean)**2) for group in groups)

        eta_squared = ss_between / ss_total

        # Degrees of freedom
        df_between = len(groups) - 1
        df_within = sum(len(group) for group in groups) - len(groups)

        # Confidence interval for F-statistic (approximate)
        f_critical_lower = stats.f.ppf((1 - self.confidence_level) / 2, df_between, df_within)
        f_critical_upper = stats.f.ppf((1 + self.confidence_level) / 2, df_between, df_within)

        # Interpret effect size
        interpretation = self._interpret_eta_squared(eta_squared)

        sample_size = sum(len(group) for group in groups)

        return StatisticalResult(
            test_name='One-way ANOVA',
            statistic=f_stat,
            p_value=p_value,
            effect_size=eta_squared,
            confidence_interval=(f_critical_lower, f_critical_upper),
            is_significant=p_value < self.alpha,
            interpretation=interpretation,
            sample_size=sample_size,
            power=None  # Power calculation complex for ANOVA
        )

    def _interpret_eta_squared(self, eta_squared: float) -> str:
        """Interpret eta-squared effect size."""
        if eta_squared < 0.01:
            return "negligible"
        elif eta_squared < 0.06:
            return "small"
        elif eta_squared < 0.14:
            return "medium"
        else:
            return "large"

    def correlation_analysis(self,
                           x: np.ndarray,
                           y: np.ndarray,
                           method: str = 'pearson') -> StatisticalResult:
        """Perform correlation analysis."""
        if method == 'pearson':
            correlation, p_value = stats.pearsonr(x, y)
        elif method == 'spearman':
            correlation, p_value = stats.spearmanr(x, y)
        elif method == 'kendall':
            correlation, p_value = stats.kendalltau(x, y)
        else:
            raise ValueError(f"Unknown correlation method: {method}")

        # Confidence interval for correlation
        n = len(x)
        if method == 'pearson':
            # Fisher z-transformation
            z = np.arctanh(correlation)
            se_z = 1 / np.sqrt(n - 3)
            z_critical = stats.norm.ppf((1 + self.confidence_level) / 2)

            z_lower = z - z_critical * se_z
            z_upper = z + z_critical * se_z

            ci_lower = np.tanh(z_lower)
            ci_upper = np.tanh(z_upper)
        else:
            # Bootstrap for non-parametric correlations
            ci_lower, ci_upper = self._bootstrap_correlation_ci(x, y, method, self.confidence_level)

        # Interpret correlation strength
        interpretation = self._interpret_correlation(correlation)

        return StatisticalResult(
            test_name=f'{method.capitalize()} correlation',
            statistic=correlation,
            p_value=p_value,
            effect_size=abs(correlation),
            confidence_interval=(ci_lower, ci_upper),
            is_significant=p_value < self.alpha,
            interpretation=interpretation,
            sample_size=n,
            power=None
        )

    def _bootstrap_correlation_ci(self,
                                 x: np.ndarray,
                                 y: np.ndarray,
                                 method: str,
                                 confidence_level: float) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for correlation."""
        n_bootstrap = 1000
        correlations = []

        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(len(x), size=len(x), replace=True)
            x_boot = x[indices]
            y_boot = y[indices]

            # Calculate correlation
            if method == 'spearman':
                corr, _ = stats.spearmanr(x_boot, y_boot)
            elif method == 'kendall':
                corr, _ = stats.kendalltau(x_boot, y_boot)
            else:
                corr, _ = stats.pearsonr(x_boot, y_boot)

            correlations.append(corr)

        # Calculate confidence interval
        alpha_level = 1 - confidence_level
        ci_lower = np.percentile(correlations, 100 * alpha_level / 2)
        ci_upper = np.percentile(correlations, 100 * (1 - alpha_level / 2))

        return ci_lower, ci_upper

    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret correlation strength."""
        abs_corr = abs(correlation)
        if abs_corr < 0.1:
            return "negligible"
        elif abs_corr < 0.3:
            return "small"
        elif abs_corr < 0.5:
            return "medium"
        elif abs_corr < 0.7:
            return "large"
        else:
            return "very large"

    def meta_analysis(self, effect_sizes: List[float],
                     sample_sizes: List[int]) -> Dict[str, Any]:
        """Perform meta-analysis to combine effect sizes from multiple studies."""
        effect_sizes = np.array(effect_sizes)
        sample_sizes = np.array(sample_sizes)

        # Calculate weights (inverse variance weighting)
        weights = sample_sizes - 3  # Approximate for correlation coefficients
        weights = weights / np.sum(weights)

        # Weighted mean effect size
        meta_effect_size = np.sum(weights * effect_sizes)

        # Standard error
        meta_se = np.sqrt(np.sum(weights**2 * (1 / (sample_sizes - 3))))

        # Confidence interval
        z_critical = stats.norm.ppf((1 + self.confidence_level) / 2)
        ci_lower = meta_effect_size - z_critical * meta_se
        ci_upper = meta_effect_size + z_critical * meta_se

        # Test for overall effect
        z_stat = meta_effect_size / meta_se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        # Test for heterogeneity (Q-statistic)
        q_stat = np.sum(weights * (effect_sizes - meta_effect_size)**2)
        df_q = len(effect_sizes) - 1
        q_p_value = 1 - stats.chi2.cdf(q_stat, df_q)

        # I² statistic for heterogeneity
        i_squared = max(0, (q_stat - df_q) / q_stat) * 100

        return {
            'meta_effect_size': meta_effect_size,
            'standard_error': meta_se,
            'confidence_interval': (ci_lower, ci_upper),
            'z_statistic': z_stat,
            'p_value': p_value,
            'is_significant': p_value < self.alpha,
            'heterogeneity': {
                'q_statistic': q_stat,
                'q_p_value': q_p_value,
                'i_squared': i_squared,
                'interpretation': 'high' if i_squared > 75 else 'moderate' if i_squared > 50 else 'low'
            },
            'number_of_studies': len(effect_sizes),
            'total_sample_size': np.sum(sample_sizes)
        }

    def multiple_comparisons_correction(self,
                                      p_values: List[float],
                                      method: str = 'bonferroni') -> Dict[str, Any]:
        """Apply multiple comparisons correction to p-values."""
        p_values = np.array(p_values)

        if method == 'bonferroni':
            corrected_alpha = self.alpha / len(p_values)
            corrected_p_values = p_values * len(p_values)
            corrected_p_values = np.minimum(corrected_p_values, 1.0)
        elif method == 'holm':
            # Holm-Bonferroni method
            sorted_indices = np.argsort(p_values)
            corrected_p_values = np.zeros_like(p_values)

            for i, idx in enumerate(sorted_indices):
                correction_factor = len(p_values) - i
                corrected_p_values[idx] = min(p_values[idx] * correction_factor, 1.0)

            corrected_alpha = self.alpha
        elif method == 'fdr_bh':
            # Benjamini-Hochberg FDR control
            sorted_indices = np.argsort(p_values)
            corrected_p_values = np.zeros_like(p_values)

            for i, idx in enumerate(sorted_indices):
                correction_factor = len(p_values) / (i + 1)
                corrected_p_values[idx] = min(p_values[idx] * correction_factor, 1.0)

            corrected_alpha = self.alpha
        else:
            raise ValueError(f"Unknown correction method: {method}")

        significant_after_correction = corrected_p_values < corrected_alpha

        return {
            'method': method,
            'original_p_values': p_values,
            'corrected_p_values': corrected_p_values,
            'corrected_alpha': corrected_alpha,
            'significant_before_correction': np.sum(p_values < self.alpha),
            'significant_after_correction': np.sum(significant_after_correction),
            'rejection_indices': np.where(significant_after_correction)[0].tolist()
        }
