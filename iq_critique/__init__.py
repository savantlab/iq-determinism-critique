"""
IQ Determinism Critique - A research library for analyzing probabilistic-to-deterministic 
transformations in psychometric testing.
"""

__version__ = "0.1.0"
__author__ = "Savant Lab"

from .models import (
    simulate_test_scores,
    calculate_confidence_interval,
    calculate_sem,
)

__all__ = [
    "simulate_test_scores",
    "calculate_confidence_interval",
    "calculate_sem",
]
