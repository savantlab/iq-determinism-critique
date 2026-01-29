# IQ Determinism Critique

A Python library for investigating how probabilistic test measurements are transformed into deterministic IQ scores, and the epistemological implications of this transformation.

## Research Objectives

1. **Probabilistic-to-Deterministic Transformation**: Analyze how measurement uncertainty in psychometric testing is collapsed into single-point IQ values
2. **Statistical Assumptions**: Examine the validity of normality assumptions, standard error calculations, and confidence intervals
3. **Measurement Theory**: Investigate Classical Test Theory (CTT) vs. Item Response Theory (IRT) frameworks
4. **Reification Critique**: Document how probabilistic measurements become treated as fixed traits

## Core Hypothesis

IQ scores are derived from probabilistic test performance but are subsequently treated as deterministic properties of individuals, obscuring:
- Measurement error (SEM typically ±3-5 points)
- Test-retest variability
- Conditional standard errors
- Practice effects and regression to the mean

## Features

- **Literature Scraping**: Collect research on IQ testing methodology, psychometrics, and critiques
- **Statistical Analysis**: Simulate test score distributions and analyze transformation effects
- **Measurement Error Modeling**: Calculate and visualize confidence intervals typically omitted from reports
- **Citation Tracking**: Curated reading list of critical psychometrics literature

## Installation

### Development Setup

```bash
git clone git@github.com:savantlab/iq-determinism-critique.git
cd iq-determinism-critique
pip install -e .
```

## Usage

### Command-Line Interface

```bash
# Scrape academic literature
iq-critique-scrape --query "IQ measurement error" --year-start 1980

# Analyze psychometric data
iq-critique-analyze --simulate-sem --iq-mean 100 --sem 5

# Manage reading list
iq-critique-reading list
```

### Python Library

```python
from iq_critique.models import simulate_test_scores, calculate_confidence_interval
from iq_critique.analysis import analyze_determinism_gap

# Simulate test administration with measurement error
scores = simulate_test_scores(true_iq=115, sem=5, n_tests=100)

# Analyze how single-point estimates mask uncertainty
analysis = analyze_determinism_gap(scores)
```

## Project Structure

```
iq-determinism-critique/
├── iq_critique/              # Python package
│   ├── __init__.py
│   ├── cli.py               # Command-line interface
│   ├── models.py            # Statistical models
│   └── analysis.py          # Analysis tools
├── scripts/                 # Research scripts
│   ├── scrape_literature.py
│   ├── simulate_tests.py
│   └── analyze_reification.py
├── data/                    # Collected data
├── notebooks/              # Jupyter analysis
├── results/                # Outputs
└── reading_list.json       # Curated papers
```

## Key Research Areas

### 1. Measurement Error
- Standard Error of Measurement (SEM)
- Confidence intervals (90%, 95%)
- Conditional SEM across ability levels

### 2. Statistical Assumptions
- Normality assumption validity
- Flynn effect and norm obsolescence
- Regression to the mean

### 3. Reification Process
- How probability distributions become point estimates
- Institutional use of single-number scores
- Policy implications of deterministic interpretation

## Literature Focus

- Gould, S.J. (1981). *The Mismeasure of Man*
- Nunnally & Bernstein. *Psychometric Theory*
- Lord & Novick. *Statistical Theories of Mental Test Scores*
- Critical psychometrics literature on IQ testing limitations

## Contributing

This is a research project maintained by Savant Lab. Contributions and discussions welcome.

## License

MIT License
