# Quickstart Guide

Get up and running with the IQ Determinism Critique library in 5 minutes.

## Installation

```bash
git clone [repository-url]
cd iq-determinism-critique
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

## Basic Usage

### 1. Run the Demonstration

See the reification problem in action:

```bash
python examples/demonstrate_reification.py
```

This will:
- Simulate someone taking an IQ test 50 times
- Show how scores vary due to measurement error (±5 points typically)
- Demonstrate the gap between probabilistic reality and deterministic reporting
- Generate visualizations in `results/`

### 2. Python API

```python
from iq_critique.models import (
    simulate_test_scores,
    calculate_confidence_interval,
    calculate_sem,
)

# Calculate typical measurement error for IQ tests
sem = calculate_sem(reliability=0.90)  # ~4.7 points
print(f"Standard Error: {sem:.2f}")

# Simulate someone with IQ 120 taking test 100 times
scores = simulate_test_scores(true_score=120, sem=sem, n_tests=100)

# What SHOULD be reported (but usually isn't)
ci = calculate_confidence_interval(scores[0], sem, confidence_level=0.95)
print(f"Reported: {scores[0]:.1f}")
print(f"95% CI: [{ci[0]:.1f}, {ci[1]:.1f}]")
```

### 3. Key Concepts

**Standard Error of Measurement (SEM)**
- Typical IQ tests: ±4-5 points
- Formula: SEM = SD × √(1 - reliability)
- For reliability=0.90, SD=15: SEM ≈ 4.74

**Confidence Intervals**
- 68% CI: ±1 SEM (~±5 points)
- 95% CI: ±2 SEM (~±10 points)
- These are usually NOT reported with IQ scores

**The Reification Problem**
1. Tests measure probabilistically (with error)
2. Scores vary across administrations
3. Single number gets reported as "THE IQ"
4. Institutions treat it as deterministic property

## Research Questions

1. How does measurement error affect classification decisions?
   - e.g., "gifted" cutoff at 130 when SEM = 5
   
2. What percentage of variance is measurement error vs. true differences?

3. How do confidence intervals change policy implications?

## Next Steps

- Read `README.md` for full documentation
- Explore `notebooks/` for detailed analysis
- Check `reading_list.json` for foundational literature
- See `scripts/` for data collection tools

## Contributing

This is an open research project. Questions, critiques, and contributions welcome!
