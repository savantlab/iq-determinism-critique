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

from .pdf_analysis import (
    plot_score_pdf,
    compare_overlapping_pdfs,
    probability_of_reversal,
    calculate_information_loss,
)

__all__ = [
    "simulate_test_scores",
    "calculate_confidence_interval",
    "calculate_sem",
    "plot_score_pdf",
    "compare_overlapping_pdfs",
    "probability_of_reversal",
    "calculate_information_loss",
]
