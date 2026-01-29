"""
Probability Density Function (PDF) analysis for IQ testing.

This module demonstrates how the continuous probability distribution
over true scores gets collapsed into deterministic point estimates.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, Optional
import seaborn as sns


def plot_score_pdf(
    observed_score: float,
    sem: float = 5.0,
    confidence_levels: list = [0.68, 0.95],
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot the PDF of true scores given an observed score.
    
    This shows what an IQ score ACTUALLY represents: a probability
    distribution over possible true values, not a single point.
    
    Args:
        observed_score: The reported IQ score
        sem: Standard error of measurement
        confidence_levels: List of confidence levels to show
        ax: Matplotlib axes (created if None)
    
    Returns:
        Matplotlib axes with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate PDF
    true_scores = np.linspace(observed_score - 4*sem, observed_score + 4*sem, 1000)
    pdf = stats.norm.pdf(true_scores, loc=observed_score, scale=sem)
    
    # Plot PDF
    ax.plot(true_scores, pdf, linewidth=2.5, color='blue', label='PDF of true score')
    ax.fill_between(true_scores, pdf, alpha=0.2, color='blue')
    
    # Mark observed score (what gets reported)
    ax.axvline(observed_score, color='red', linestyle='--', linewidth=2,
               label=f'Reported: {observed_score}')
    
    # Show confidence intervals
    colors = ['orange', 'green', 'purple']
    for i, conf in enumerate(confidence_levels):
        z = stats.norm.ppf((1 + conf) / 2)
        lower = observed_score - z * sem
        upper = observed_score + z * sem
        ax.axvspan(lower, upper, alpha=0.15, color=colors[i % len(colors)],
                   label=f'{conf*100:.0f}% CI: [{lower:.1f}, {upper:.1f}]')
    
    ax.set_xlabel('True Score', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title(f'PDF of True Score | Observed = {observed_score} (SEM = {sem})',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    return ax


def compare_overlapping_pdfs(
    scores: list,
    sem: float = 5.0,
    labels: Optional[list] = None,
    cutoff: Optional[float] = None,
) -> plt.Figure:
    """
    Compare PDFs for multiple individuals to show overlap.
    
    This demonstrates why binary classifications (pass/fail, gifted/not)
    are problematic when PDFs overlap significantly.
    
    Args:
        scores: List of observed scores
        sem: Standard error of measurement
        labels: Labels for each person
        cutoff: Optional classification cutoff to show
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    if labels is None:
        labels = [f'Score {s}' for s in scores]
    
    # Determine plot range
    min_score = min(scores) - 3*sem
    max_score = max(scores) + 3*sem
    true_scores = np.linspace(min_score, max_score, 1000)
    
    # Plot each PDF
    colors = sns.color_palette("husl", len(scores))
    
    for score, label, color in zip(scores, labels, colors):
        pdf = stats.norm.pdf(true_scores, loc=score, scale=sem)
        ax.plot(true_scores, pdf, label=label, linewidth=2.5, color=color)
        ax.axvline(score, color=color, linestyle=':', alpha=0.5)
    
    # Show cutoff if provided
    if cutoff is not None:
        ax.axvline(cutoff, color='black', linestyle='--', linewidth=2,
                   label=f'Classification cutoff: {cutoff}')
        
        # Calculate probabilities of being above cutoff
        for score, label in zip(scores, labels):
            prob_above = 1 - stats.norm.cdf(cutoff, loc=score, scale=sem)
            print(f"{label}: P(true score > {cutoff}) = {prob_above:.1%}")
    
    ax.set_xlabel('True Score', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title('Overlapping PDFs: Why Rankings Are Uncertain', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def probability_of_reversal(
    score_a: float,
    score_b: float,
    sem: float = 5.0,
) -> dict:
    """
    Calculate probability that person A's true score exceeds person B's,
    even though B scored higher on the test.
    
    This quantifies how measurement error creates ranking uncertainty.
    
    Args:
        score_a: First observed score
        score_b: Second observed score
        sem: Standard error of measurement
    
    Returns:
        Dictionary with reversal probability and related statistics
    """
    # Difference of two normal distributions is also normal
    diff_mean = score_a - score_b
    diff_sd = np.sqrt(2 * sem**2)
    
    # P(A > B) = P(A - B > 0)
    prob_a_greater = 1 - stats.norm.cdf(0, loc=diff_mean, scale=diff_sd)
    
    return {
        'score_a': score_a,
        'score_b': score_b,
        'difference': score_b - score_a,
        'prob_a_greater_than_b': prob_a_greater,
        'prob_b_greater_than_a': 1 - prob_a_greater,
        'sem': sem,
        'note': 'Probability assumes independent measurement errors'
    }


def calculate_information_loss(
    observed_score: float,
    sem: float = 5.0,
) -> dict:
    """
    Quantify information lost when collapsing PDF to point estimate.
    
    The PDF contains rich information about uncertainty. Point estimates
    discard nearly all of this.
    
    Args:
        observed_score: The reported score
        sem: Standard error of measurement
    
    Returns:
        Dictionary showing information in PDF vs. point estimate
    """
    # PDF information (entropy)
    # For normal distribution: H = 0.5 * log(2πe * σ²)
    pdf_entropy = 0.5 * np.log(2 * np.pi * np.e * sem**2)
    
    # Point estimate information: 0 (no uncertainty represented)
    point_entropy = 0
    
    # Information loss
    info_loss = pdf_entropy - point_entropy
    
    # Equivalent width of uniform distribution with same entropy
    equiv_width = np.sqrt(2 * np.pi * np.e) * sem
    
    return {
        'observed_score': observed_score,
        'sem': sem,
        'pdf_entropy_nats': pdf_entropy,
        'point_estimate_entropy': point_entropy,
        'information_loss_nats': info_loss,
        'information_loss_bits': info_loss / np.log(2),
        'equivalent_uniform_width': equiv_width,
        '68_percent_interval_width': 2 * sem,
        '95_percent_interval_width': 2 * 1.96 * sem,
    }


def demonstrate_pdf_height_vs_probability():
    """
    Educational demo: PDF height is NOT probability.
    
    This is a common misunderstanding. PDF(x) can exceed 1!
    Probability requires integrating over an interval.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Example 1: PDF height can exceed 1
    sem_small = 2  # Small SEM → tall, narrow PDF
    observed = 115
    
    scores = np.linspace(105, 125, 1000)
    pdf = stats.norm.pdf(scores, loc=observed, scale=sem_small)
    
    ax1.plot(scores, pdf, linewidth=2)
    ax1.axhline(1.0, color='red', linestyle='--', label='y = 1.0')
    ax1.fill_between(scores, pdf, alpha=0.3)
    max_pdf = np.max(pdf)
    ax1.text(observed, max_pdf, f'Max PDF: {max_pdf:.3f}\n(> 1.0!)', 
             ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Score')
    ax1.set_ylabel('PDF(x)')
    ax1.set_title(f'PDF Height Can Exceed 1.0\n(SEM = {sem_small})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Example 2: Probability requires integration
    sem = 5
    scores = np.linspace(90, 140, 1000)
    pdf = stats.norm.pdf(scores, loc=observed, scale=sem)
    
    # Show that integral over small interval gives probability
    interval_width = 1
    mask = (scores >= observed - interval_width/2) & (scores <= observed + interval_width/2)
    
    ax2.plot(scores, pdf, linewidth=2, label='PDF')
    ax2.fill_between(scores[mask], pdf[mask], alpha=0.5, color='orange',
                     label=f'P({observed-interval_width/2} < x < {observed+interval_width/2})')
    
    prob = stats.norm.cdf(observed + interval_width/2, loc=observed, scale=sem) - \
           stats.norm.cdf(observed - interval_width/2, loc=observed, scale=sem)
    
    ax2.text(observed, np.max(pdf)/2, 
             f'Shaded area\n= {prob:.4f}\n(probability)', 
             ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat'))
    ax2.set_xlabel('Score')
    ax2.set_ylabel('PDF(x)')
    ax2.set_title(f'Probability = Area Under PDF\n(SEM = {sem})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def mixture_distribution_analysis(
    n_individuals: int = 1000,
    population_mean: float = 100,
    population_sd: float = 15,
    sem: float = 5,
) -> dict:
    """
    Show how individual measurement PDFs create a mixture distribution
    that is wider than the true population distribution.
    
    This demonstrates: Var(observed) = Var(true) + Var(error)
    
    Args:
        n_individuals: Number of people in simulation
        population_mean: True population mean
        population_sd: True population SD
        sem: Standard error of measurement
    
    Returns:
        Dictionary with analysis results and figure
    """
    # Sample true scores from population
    true_scores = np.random.normal(population_mean, population_sd, n_individuals)
    
    # Add measurement error
    observed_scores = true_scores + np.random.normal(0, sem, n_individuals)
    
    # Calculate observed variance
    observed_sd = np.std(observed_scores)
    theoretical_observed_sd = np.sqrt(population_sd**2 + sem**2)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Histogram of observed scores
    ax.hist(observed_scores, bins=50, density=True, alpha=0.5, 
            label=f'Observed (SD={observed_sd:.2f})', edgecolor='black')
    
    # Overlay true population distribution
    x = np.linspace(40, 160, 1000)
    true_pdf = stats.norm.pdf(x, loc=population_mean, scale=population_sd)
    ax.plot(x, true_pdf, 'g-', linewidth=2.5, 
            label=f'True population (SD={population_sd})')
    
    # Overlay theoretical observed distribution
    observed_pdf = stats.norm.pdf(x, loc=population_mean, scale=theoretical_observed_sd)
    ax.plot(x, observed_pdf, 'r--', linewidth=2.5, 
            label=f'Theoretical observed (SD={theoretical_observed_sd:.2f})')
    
    ax.set_xlabel('IQ Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Measurement Error Inflates Observed Variance\n' + 
                 r'$\sigma_{observed}^2 = \sigma_{true}^2 + \sigma_{error}^2$',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    return {
        'true_population_sd': population_sd,
        'sem': sem,
        'theoretical_observed_sd': theoretical_observed_sd,
        'empirical_observed_sd': observed_sd,
        'variance_inflation': (observed_sd**2 - population_sd**2) / population_sd**2,
        'figure': fig,
    }
