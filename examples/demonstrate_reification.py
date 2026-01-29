#!/usr/bin/env python3
"""
Demonstration: How probabilistic test scores become deterministic IQ values.

This example shows:
1. A person takes an IQ test multiple times
2. Scores vary due to measurement error (SEM ≈ 5 points)
3. One score gets reported as "THE IQ"
4. Confidence intervals are omitted
"""

import numpy as np
import matplotlib.pyplot as plt
from iq_critique.models import (
    simulate_test_scores,
    calculate_confidence_interval,
    calculate_sem,
    analyze_reification,
)


def main():
    # Typical IQ test parameters
    RELIABILITY = 0.90  # Typical for major IQ tests (WAIS, Stanford-Binet)
    SEM = calculate_sem(RELIABILITY, sd=15)
    print(f"Standard Error of Measurement: {SEM:.2f} points")
    print(f"(This means 68% of scores fall within ±{SEM:.2f} of true score)\n")
    
    # Simulate someone with "true score" of 115 taking the test 50 times
    TRUE_SCORE = 115
    N_TESTS = 50
    
    scores = simulate_test_scores(true_score=TRUE_SCORE, sem=SEM, n_tests=N_TESTS)
    
    # What gets reported: just the first test
    reported = scores[0]
    ci_95 = calculate_confidence_interval(reported, SEM, 0.95)
    
    print("=" * 60)
    print("SIMULATION: Person takes IQ test 50 times")
    print("=" * 60)
    print(f"True underlying ability: {TRUE_SCORE}")
    print(f"Reported IQ (first test): {reported:.1f}")
    print(f"95% Confidence Interval: [{ci_95[0]:.1f}, {ci_95[1]:.1f}]")
    print(f"\nActual scores across 50 tests:")
    print(f"  Mean: {np.mean(scores):.1f}")
    print(f"  Range: {np.min(scores):.1f} to {np.max(scores):.1f}")
    print(f"  Std Dev: {np.std(scores):.2f}")
    print()
    
    # Analysis
    analysis = analyze_reification(scores, reported)
    print("=" * 60)
    print("THE REIFICATION PROBLEM")
    print("=" * 60)
    print("What gets reported: A single number (e.g., 'IQ = 115')")
    print("What actually happened: Probabilistic measurement with error")
    print()
    print(f"If tested again, expected range (95% CI): "
          f"[{analysis['ci_95_lower']:.1f}, {analysis['ci_95_upper']:.1f}]")
    print(f"Actual observed range: "
          f"[{analysis['min_score']:.1f}, {analysis['max_score']:.1f}]")
    print()
    print("Yet institutions treat the reported score as deterministic.")
    
    # Visualization
    plt.figure(figsize=(12, 6))
    
    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(scores, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(reported, color='red', linestyle='--', linewidth=2, 
                label=f'Reported: {reported:.1f}')
    plt.axvline(TRUE_SCORE, color='green', linestyle='--', linewidth=2,
                label=f'True: {TRUE_SCORE}')
    plt.axvline(ci_95[0], color='orange', linestyle=':', linewidth=1.5)
    plt.axvline(ci_95[1], color='orange', linestyle=':', linewidth=1.5,
                label='95% CI')
    plt.xlabel('IQ Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Observed Scores\n(50 test administrations)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Timeline
    plt.subplot(1, 2, 2)
    plt.plot(range(1, N_TESTS + 1), scores, 'o-', alpha=0.6)
    plt.axhline(reported, color='red', linestyle='--', linewidth=2,
                label=f'Reported (test #1): {reported:.1f}')
    plt.axhline(TRUE_SCORE, color='green', linestyle='--', linewidth=2,
                label=f'True score: {TRUE_SCORE}')
    plt.fill_between(range(1, N_TESTS + 1), ci_95[0], ci_95[1],
                     alpha=0.2, color='orange', label='95% CI')
    plt.xlabel('Test Administration')
    plt.ylabel('IQ Score')
    plt.title('Score Variability Across Tests\n(Same person, same ability)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/reification_demo.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: results/reification_demo.png")
    

if __name__ == "__main__":
    import os
    os.makedirs('results', exist_ok=True)
    main()
