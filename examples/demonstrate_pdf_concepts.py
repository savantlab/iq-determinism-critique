#!/usr/bin/env python3
"""
Demonstration: Probability Density Functions in IQ Testing

This script shows:
1. How IQ scores are actually PDFs, not point estimates
2. Why PDF height ≠ probability
3. How overlapping PDFs make classification uncertain
4. Information loss when collapsing PDF to single number
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from iq_critique.pdf_analysis import (
    plot_score_pdf,
    compare_overlapping_pdfs,
    probability_of_reversal,
    calculate_information_loss,
    demonstrate_pdf_height_vs_probability,
    mixture_distribution_analysis,
)


def main():
    os.makedirs('results/pdf_analysis', exist_ok=True)
    
    print("=" * 70)
    print("PDF ANALYSIS: What IQ Scores ACTUALLY Mean")
    print("=" * 70)
    print()
    
    # Example 1: Single score PDF
    print("1. A single IQ score is a DISTRIBUTION, not a point")
    print("-" * 70)
    
    observed = 115
    sem = 5
    
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_score_pdf(observed, sem, ax=ax)
    plt.savefig('results/pdf_analysis/single_score_pdf.png', dpi=150, bbox_inches='tight')
    print(f"Observed score: {observed}")
    print(f"What this ACTUALLY means: True score ∈ [{observed-sem:.1f}, {observed+sem:.1f}] (68% CI)")
    print(f"                         True score ∈ [{observed-1.96*sem:.1f}, {observed+1.96*sem:.1f}] (95% CI)")
    print(f"Saved: results/pdf_analysis/single_score_pdf.png")
    print()
    
    # Example 2: Overlapping PDFs
    print("2. Overlapping PDFs make rankings uncertain")
    print("-" * 70)
    
    scores = [110, 120, 125]
    labels = ['Person A (110)', 'Person B (120)', 'Person C (125)']
    
    fig = compare_overlapping_pdfs(scores, sem=5, labels=labels, cutoff=130)
    plt.savefig('results/pdf_analysis/overlapping_pdfs.png', dpi=150, bbox_inches='tight')
    print(f"Saved: results/pdf_analysis/overlapping_pdfs.png")
    print()
    
    # Example 3: Ranking reversal probability
    print("3. Probability of ranking reversal")
    print("-" * 70)
    
    result = probability_of_reversal(score_a=118, score_b=125, sem=5)
    print(f"Person A scored: {result['score_a']}")
    print(f"Person B scored: {result['score_b']}")
    print(f"Difference: {result['difference']} points")
    print(f"\nP(A's true score > B's true score): {result['prob_a_greater_than_b']:.1%}")
    print(f"P(B's true score > A's true score): {result['prob_b_greater_than_a']:.1%}")
    print("\nEven though B scored 7 points higher, there's a non-trivial chance")
    print("that A's true ability exceeds B's!")
    print()
    
    # Example 4: Information loss
    print("4. Information lost when reporting only a point estimate")
    print("-" * 70)
    
    info = calculate_information_loss(observed_score=115, sem=5)
    print(f"Observed score: {info['observed_score']}")
    print(f"SEM: {info['sem']}")
    print(f"\nPDF entropy: {info['pdf_entropy_nats']:.3f} nats ({info['information_loss_bits']:.3f} bits)")
    print(f"Point estimate entropy: {info['point_estimate_entropy']} (no uncertainty)")
    print(f"\nInformation loss: {info['information_loss_bits']:.3f} bits")
    print(f"\nEquivalent uncertainty (uniform distribution): ±{info['equivalent_uniform_width']/2:.2f} points")
    print(f"95% interval that SHOULD be reported: ±{info['95_percent_interval_width']/2:.2f} points")
    print()
    
    # Example 5: PDF height vs. probability
    print("5. Common misconception: PDF height ≠ probability")
    print("-" * 70)
    
    fig = demonstrate_pdf_height_vs_probability()
    plt.savefig('results/pdf_analysis/pdf_height_vs_probability.png', dpi=150, bbox_inches='tight')
    print("PDF values can exceed 1.0 (unlike probabilities)")
    print("Probability requires integrating (area under curve)")
    print(f"Saved: results/pdf_analysis/pdf_height_vs_probability.png")
    print()
    
    # Example 6: Mixture distribution
    print("6. Measurement error inflates observed variance")
    print("-" * 70)
    
    result = mixture_distribution_analysis(n_individuals=1000, 
                                          population_mean=100,
                                          population_sd=15,
                                          sem=5)
    
    print(f"True population SD: {result['true_population_sd']}")
    print(f"SEM: {result['sem']}")
    print(f"Theoretical observed SD: {result['theoretical_observed_sd']:.2f}")
    print(f"Empirical observed SD: {result['empirical_observed_sd']:.2f}")
    print(f"\nVariance inflation: {result['variance_inflation']:.1%}")
    print(f"\nFormula: σ²_observed = σ²_true + σ²_error")
    print(f"         {result['theoretical_observed_sd']**2:.1f} = {result['true_population_sd']**2} + {result['sem']**2}")
    
    result['figure'].savefig('results/pdf_analysis/mixture_distribution.png', 
                            dpi=150, bbox_inches='tight')
    print(f"Saved: results/pdf_analysis/mixture_distribution.png")
    print()
    
    # Summary
    print("=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("1. IQ scores are PROBABILITY DISTRIBUTIONS, not points")
    print("2. PDFs contain uncertainty information that gets discarded")
    print("3. Overlapping PDFs make binary classifications problematic")
    print("4. Measurement error inflates observed population variance")
    print("5. Reporting only point estimates loses ~3 bits of information")
    print()
    print("All visualizations saved to: results/pdf_analysis/")
    print()
    
    plt.show()


if __name__ == "__main__":
    main()
