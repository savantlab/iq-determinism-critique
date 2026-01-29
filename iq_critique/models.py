"""
Statistical models for analyzing IQ test measurements and their transformation
from probabilistic scores to deterministic values.
"""

import numpy as np
from scipy import stats
from typing import Tuple, Dict, Optional
import pandas as pd


def calculate_sem(reliability: float, sd: float = 15) -> float:
    """
    Calculate Standard Error of Measurement.
    
    SEM = SD * sqrt(1 - reliability)
    
    Args:
        reliability: Test reliability coefficient (0-1)
        sd: Standard deviation of test (typically 15 for IQ tests)
    
    Returns:
        Standard error of measurement
    """
    return sd * np.sqrt(1 - reliability)


def simulate_test_scores(
    true_score: float,
    sem: float,
    n_tests: int = 100,
    mean: float = 100,
    sd: float = 15,
) -> np.ndarray:
    """
    Simulate repeated test administrations with measurement error.
    
    In Classical Test Theory:
        Observed Score = True Score + Error
        where Error ~ N(0, SEM²)
    
    Args:
        true_score: Individual's "true" ability (if it existed)
        sem: Standard error of measurement
        n_tests: Number of test administrations to simulate
        mean: Population mean (typically 100)
        sd: Population standard deviation (typically 15)
    
    Returns:
        Array of simulated observed scores
    """
    # Add measurement error to true score
    errors = np.random.normal(0, sem, n_tests)
    observed_scores = true_score + errors
    
    # Clip to realistic range (typically 40-160)
    return np.clip(observed_scores, mean - 4*sd, mean + 4*sd)


def calculate_confidence_interval(
    observed_score: float,
    sem: float,
    confidence_level: float = 0.95,
) -> Tuple[float, float]:
    """
    Calculate confidence interval for true score given observed score.
    
    This is what IQ reports should include but typically don't.
    
    Args:
        observed_score: The reported IQ score
        sem: Standard error of measurement
        confidence_level: Confidence level (0.90, 0.95, 0.99)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    margin = z_score * sem
    return (observed_score - margin, observed_score + margin)


def analyze_reification(
    scores: np.ndarray,
    reported_score: Optional[float] = None,
) -> Dict[str, float]:
    """
    Analyze the gap between probabilistic reality and deterministic reporting.
    
    Args:
        scores: Array of simulated test scores
        reported_score: The single number that would be reported (default: mean)
    
    Returns:
        Dictionary with analysis metrics
    """
    if reported_score is None:
        reported_score = np.mean(scores)
    
    sem_empirical = np.std(scores, ddof=1)
    ci_90 = np.percentile(scores, [5, 95])
    ci_95 = np.percentile(scores, [2.5, 97.5])
    
    return {
        "reported_score": reported_score,
        "mean_observed": np.mean(scores),
        "std_observed": np.std(scores),
        "sem_empirical": sem_empirical,
        "ci_90_lower": ci_90[0],
        "ci_90_upper": ci_90[1],
        "ci_95_lower": ci_95[0],
        "ci_95_upper": ci_95[1],
        "range": np.ptp(scores),
        "min_score": np.min(scores),
        "max_score": np.max(scores),
    }


def simulate_multiple_individuals(
    n_individuals: int = 100,
    reliability: float = 0.90,
    mean: float = 100,
    sd: float = 15,
    n_tests_per_person: int = 10,
) -> pd.DataFrame:
    """
    Simulate testing multiple individuals to show population-level reification.
    
    Args:
        n_individuals: Number of people to simulate
        reliability: Test reliability
        mean: Population mean
        sd: Population standard deviation
        n_tests_per_person: How many times each person takes the test
    
    Returns:
        DataFrame with individual results
    """
    sem = calculate_sem(reliability, sd)
    results = []
    
    for i in range(n_individuals):
        # Sample a "true score" from the population
        true_score = np.random.normal(mean, sd)
        
        # Simulate multiple test administrations
        observed_scores = simulate_test_scores(true_score, sem, n_tests_per_person)
        
        # What gets reported vs. reality
        reported = observed_scores[0]  # First test score
        ci_95 = calculate_confidence_interval(reported, sem, 0.95)
        
        results.append({
            "individual_id": i,
            "true_score": true_score,
            "reported_score": reported,
            "mean_across_tests": np.mean(observed_scores),
            "std_across_tests": np.std(observed_scores),
            "ci_95_lower": ci_95[0],
            "ci_95_upper": ci_95[1],
            "true_in_ci": ci_95[0] <= true_score <= ci_95[1],
        })
    
    return pd.DataFrame(results)


def demonstrate_regression_to_mean(
    initial_score: float,
    population_mean: float = 100,
    reliability: float = 0.90,
    sem: Optional[float] = None,
) -> Dict[str, float]:
    """
    Demonstrate regression to the mean effect in retest scenarios.
    
    Extreme scores on first test are likely to be less extreme on retest
    due to measurement error.
    
    Args:
        initial_score: First observed score
        population_mean: Population mean (100)
        reliability: Test reliability
        sem: Standard error (calculated if not provided)
    
    Returns:
        Expected retest score and related metrics
    """
    if sem is None:
        sem = calculate_sem(reliability)
    
    # Regression equation: E(X₂|X₁) = μ + r(X₁ - μ)
    expected_retest = population_mean + reliability * (initial_score - population_mean)
    regression_effect = initial_score - expected_retest
    
    return {
        "initial_score": initial_score,
        "expected_retest": expected_retest,
        "regression_effect": regression_effect,
        "sem": sem,
        "reliability": reliability,
    }
