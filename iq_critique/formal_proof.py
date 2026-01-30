"""
Formal mathematical proof that probabilistic discrete random variables
are not equal to deterministic discrete random variables.

This establishes the theoretical foundation for the IQ reification critique.
"""

import numpy as np
from scipy import stats
from typing import Dict, Callable
import matplotlib.pyplot as plt


class DiscreteRandomVariable:
    """Base class for discrete random variables."""
    
    def __init__(self, outcomes: np.ndarray, pmf: np.ndarray):
        """
        Args:
            outcomes: Array of possible outcomes
            pmf: Probability mass function (must sum to 1)
        """
        assert np.abs(np.sum(pmf) - 1.0) < 1e-10, "PMF must sum to 1"
        assert len(outcomes) == len(pmf), "Outcomes and PMF must have same length"
        assert np.all(pmf >= 0), "Probabilities must be non-negative"
        
        self.outcomes = outcomes
        self.pmf = pmf
    
    def entropy(self) -> float:
        """
        Calculate Shannon entropy: H(X) = -Σ p_i log(p_i)
        
        Returns:
            Entropy in nats (using natural log)
        """
        # Handle p=0 case (0 * log(0) = 0 by convention)
        nonzero = self.pmf > 0
        return -np.sum(self.pmf[nonzero] * np.log(self.pmf[nonzero]))
    
    def variance(self) -> float:
        """Calculate variance: Var(X) = Σ p_i(x_i - μ)²"""
        mean = np.sum(self.outcomes * self.pmf)
        return np.sum(self.pmf * (self.outcomes - mean)**2)
    
    def mean(self) -> float:
        """Calculate expected value: E[X] = Σ p_i x_i"""
        return np.sum(self.outcomes * self.pmf)
    
    def is_deterministic(self, tol: float = 1e-10) -> bool:
        """
        Check if this is a deterministic random variable.
        
        A random variable is deterministic if exactly one outcome has p=1.
        """
        return np.sum(np.abs(self.pmf - 1.0) < tol) == 1
    
    def is_probabilistic(self, tol: float = 1e-10) -> bool:
        """
        Check if this is a (non-trivial) probabilistic random variable.
        
        A random variable is probabilistic if at least two outcomes have
        probabilities strictly between 0 and 1.
        """
        intermediate = (self.pmf > tol) & (self.pmf < 1 - tol)
        return np.sum(intermediate) >= 2
    
    def plot_pmf(self, ax=None, **kwargs):
        """Plot the probability mass function."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.bar(self.outcomes, self.pmf, **kwargs)
        ax.set_xlabel('Outcome')
        ax.set_ylabel('Probability')
        ax.set_title('Probability Mass Function')
        ax.grid(True, alpha=0.3)
        
        return ax


class DeterministicRV(DiscreteRandomVariable):
    """
    Deterministic discrete random variable.
    
    Definition: P(X = x₀) = 1, P(X = x) = 0 for x ≠ x₀
    """
    
    def __init__(self, value: float):
        """
        Create a deterministic random variable.
        
        Args:
            value: The single outcome with probability 1
        """
        super().__init__(
            outcomes=np.array([value]),
            pmf=np.array([1.0])
        )
        self.value = value


class ProbabilisticRV(DiscreteRandomVariable):
    """
    Probabilistic discrete random variable.
    
    Definition: At least two outcomes have 0 < p_i < 1
    """
    
    def __init__(self, outcomes: np.ndarray, pmf: np.ndarray):
        """
        Create a probabilistic random variable.
        
        Args:
            outcomes: Array of possible outcomes
            pmf: Probability mass function
        
        Raises:
            ValueError: If the RV is deterministic
        """
        super().__init__(outcomes, pmf)
        
        if not self.is_probabilistic():
            raise ValueError(
                "Not a probabilistic RV: must have at least two "
                "outcomes with 0 < p < 1"
            )


def prove_not_equal(X: DiscreteRandomVariable, Y: DiscreteRandomVariable) -> Dict:
    """
    Prove that two random variables are not equal by showing their
    properties differ.
    
    Args:
        X: First random variable (typically deterministic)
        Y: Second random variable (typically probabilistic)
    
    Returns:
        Dictionary with proof results
    """
    # Method 1: Compare entropies
    H_X = X.entropy()
    H_Y = Y.entropy()
    entropy_differs = not np.isclose(H_X, H_Y)
    
    # Method 2: Compare variances
    var_X = X.variance()
    var_Y = Y.variance()
    variance_differs = not np.isclose(var_X, var_Y)
    
    # Method 3: Compare PMFs directly
    # Extend to common support
    all_outcomes = np.union1d(X.outcomes, Y.outcomes)
    
    pmf_X_extended = np.zeros(len(all_outcomes))
    pmf_Y_extended = np.zeros(len(all_outcomes))
    
    for i, outcome in enumerate(all_outcomes):
        if outcome in X.outcomes:
            idx = np.where(X.outcomes == outcome)[0][0]
            pmf_X_extended[i] = X.pmf[idx]
        if outcome in Y.outcomes:
            idx = np.where(Y.outcomes == outcome)[0][0]
            pmf_Y_extended[i] = Y.pmf[idx]
    
    pmf_differs = not np.allclose(pmf_X_extended, pmf_Y_extended)
    
    return {
        'X_is_deterministic': X.is_deterministic(),
        'Y_is_deterministic': Y.is_deterministic(),
        'X_is_probabilistic': X.is_probabilistic(),
        'Y_is_probabilistic': Y.is_probabilistic(),
        'entropy_X': H_X,
        'entropy_Y': H_Y,
        'entropy_differs': entropy_differs,
        'variance_X': var_X,
        'variance_Y': var_Y,
        'variance_differs': variance_differs,
        'pmf_differs': pmf_differs,
        'conclusion': 'X ≠ Y' if (entropy_differs or variance_differs or pmf_differs) else 'X = Y'
    }


def discretize_normal_pdf(mean: float, std: float, n_bins: int = 20) -> ProbabilisticRV:
    """
    Discretize a normal distribution to create a probabilistic discrete RV.
    
    This simulates what an IQ score with measurement error looks like.
    
    Args:
        mean: Mean of the normal distribution
        std: Standard deviation (e.g., SEM)
        n_bins: Number of discrete bins
    
    Returns:
        ProbabilisticRV representing discretized normal
    """
    # Create bins
    outcomes = np.linspace(mean - 4*std, mean + 4*std, n_bins)
    bin_width = outcomes[1] - outcomes[0]
    
    # Calculate probabilities (normalize to sum to 1)
    pmf = stats.norm.pdf(outcomes, loc=mean, scale=std)
    pmf = pmf / np.sum(pmf)  # Normalize
    
    return ProbabilisticRV(outcomes, pmf)


def iq_reification_proof(observed_score: float = 115, sem: float = 5) -> Dict:
    """
    Formal proof that IQ reification transforms probabilistic to deterministic.
    
    Args:
        observed_score: The reported IQ score
        sem: Standard error of measurement
    
    Returns:
        Dictionary with proof components
    """
    # What the test actually measures: probabilistic distribution
    Y_measured = discretize_normal_pdf(mean=observed_score, std=sem, n_bins=50)
    
    # What gets reported: deterministic value
    X_reported = DeterministicRV(value=observed_score)
    
    # Prove they're not equal
    proof = prove_not_equal(X_reported, Y_measured)
    
    # Visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot reported (deterministic)
    X_reported.plot_pmf(ax=ax1, color='red', alpha=0.7, edgecolor='black')
    ax1.set_title(f'Reported: Deterministic\nH(X) = {proof["entropy_X"]:.3f}')
    ax1.set_ylim([0, 1.1])
    
    # Plot measured (probabilistic)
    Y_measured.plot_pmf(ax=ax2, color='blue', alpha=0.7, edgecolor='black')
    ax2.set_title(f'Measured: Probabilistic\nH(Y) = {proof["entropy_Y"]:.3f}')
    
    # Comparison
    ax3.bar(['Reported', 'Measured'], 
            [proof['entropy_X'], proof['entropy_Y']],
            color=['red', 'blue'], alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Entropy (nats)')
    ax3.set_title('Information Content\n(Entropy ≠ ⟹ Not Equal)')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    proof['figure'] = fig
    proof['Y_measured'] = Y_measured
    proof['X_reported'] = X_reported
    
    return proof


def theorem_statement() -> str:
    """
    Return formal statement of the theorem.
    """
    return """
THEOREM: Probabilistic ≠ Deterministic

Let X be a deterministic discrete random variable and Y be a probabilistic 
discrete random variable.

DEFINITIONS:

Deterministic RV (X):
    P(X = x_k) = 1 for some k
    P(X = x_i) = 0 for all i ≠ k
    
Probabilistic RV (Y):
    ∃ at least two indices j, m such that 0 < P(Y = y_j) < 1 and 0 < P(Y = y_m) < 1

PROOF (by contradiction):

Assume X ≡ Y (they are equal).

Then their probability mass functions must be identical:
    P(X = x_i) = P(Y = y_i) for all i

From X deterministic:
    P(X = x_k) = 1, P(X = x_i) = 0 for i ≠ k

From Y probabilistic:
    ∃ j, m: 0 < P(Y = y_j) < 1 and 0 < P(Y = y_m) < 1

For X ≡ Y to hold:
    P(Y = y_j) = P(X = x_j) ∈ {0, 1}

But this contradicts 0 < P(Y = y_j) < 1.

Therefore: X ≢ Y (they are not equal)

QED

COROLLARY (Entropy Formulation):

For deterministic X:
    H(X) = 0

For probabilistic Y:
    H(Y) = -Σ p_i log(p_i) > 0

Since H(X) ≠ H(Y), the distributions are distinct.

APPLICATION TO IQ TESTING:

IQ measurement produces Y (probabilistic: true score ~ N(observed, SEM²))
IQ reporting produces X (deterministic: "your IQ is 115")

By the theorem: Y ≢ X

The reification process transforms a probabilistic measurement into a 
deterministic claim, losing information and obscuring uncertainty.
"""


if __name__ == "__main__":
    print(theorem_statement())
