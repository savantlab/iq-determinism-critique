# Research Agenda: IQ Determinism Critique

## Core Thesis

IQ tests produce **probabilistic outcomes** (scores with measurement error) that are subsequently treated as **deterministic properties, i.e. measurements** of individuals. This transformation obscures uncertainty and has significant epistemological and social implications.

## Phase 1: Foundation (Current)

### Completed ✓
- [x] Statistical modeling framework (Classical Test Theory)
- [x] Simulation tools for test-retest variability
- [x] Confidence interval calculations
- [x] Demonstration of reification problem

### In Progress
- [ ] Literature review on psychometric critiques
- [ ] Collect citations on measurement error reporting practices
- [ ] Document historical claims about IQ as fixed vs. malleable

## Phase 2: Empirical Analysis

### Measurement Error Impact
1. **Classification Errors**
   - Simulate how many people cross diagnostic thresholds due to SEM alone
   - Analyze cutoff decisions (gifted=130, ID=70, etc.) with CI overlay
   - Calculate false positive/negative rates

2. **Test-Retest Reliability**
   - Collect published reliability coefficients for major IQ tests
   - Calculate implied confidence intervals
   - Compare to reporting practices in clinical/educational settings

3. **Conditional SEM**
   - Most tests have varying SEM across ability levels
   - Investigate whether this is acknowledged in interpretation

### Data Collection
- [ ] Scrape Google Scholar for papers on "IQ measurement error"
- [ ] Collect psychometric critiques from 1980-present
- [ ] Find examples of IQ reporting (clinical reports, research papers)
- [ ] Document whether CIs are included or omitted

## Phase 3: Theoretical Analysis

### Reification Mechanisms
1. **Statistical to Ontological**
   - How does a probability distribution become a "thing"?
   - Role of language: "has an IQ of" vs. "scored approximately"
   - Institutional practices that enforce deterministic interpretation

2. **Measurement Theory**
   - Classical Test Theory: Observed = True + Error
   - The assumption of a "true score" itself is problematic
   - Alternative frameworks (Item Response Theory, latent variable models)

3. **Philosophical Implications**
   - Reification as category error
   - Operationalism and construct validity
   - The g-factor debate

### Key Questions
- Why do confidence intervals disappear in practice?
- What incentives exist to treat scores as deterministic?
- How does this affect educational policy, employment, etc.?

## Phase 4: Comparative Analysis

### Other Psychometric Domains
- Personality tests (Big Five, MMPI)
- Achievement tests (SAT, GRE)
- Clinical assessments (depression scales)

**Research question:** Is IQ uniquely treated as deterministic, or is this broader pattern?

### Historical Analysis
- How have claims about IQ changed over time?
- Spearman's g (1904) → Jensen (1969) → modern usage
- Flynn effect and norm obsolescence

## Phase 5: Visualization & Communication

### Interactive Demonstrations
- [ ] Web app: "Take this IQ test 100 times"
- [ ] Visualization of CI bands over time
- [ ] Classification boundary simulation

### Policy Implications
- [ ] White paper on educational placement decisions
- [ ] Analysis of employment testing practices
- [ ] Legal implications (disability determination, etc.)

## Key Literature to Acquire

### Psychometric Theory
- [ ] Nunnally & Bernstein - *Psychometric Theory* (1994)
- [ ] Lord & Novick - *Statistical Theories* (1968)
- [ ] Cronbach - *Essentials of Psychological Testing*

### Critiques
- [ ] Gould - *Mismeasure of Man* (1981, 1996)
- [ ] Richardson - *What IQ Tests Test*
- [ ] Sternberg - Alternative theories of intelligence

### Policy & Practice
- [ ] AERA/APA/NCME Standards (2014)
- [ ] Clinical practice guidelines
- [ ] Educational placement standards

## Methodological Notes

### Statistical Simulations
- Monte Carlo for classification errors
- Bootstrap for CI robustness
- Sensitivity analysis for reliability estimates

### Data Sources
- Google Scholar (academic critiques)
- PsycINFO (clinical literature)
- ERIC (educational applications)
- Legal databases (court cases involving IQ)

## Open Questions

1. **Empirical**: What % of IQ reports include confidence intervals?
2. **Theoretical**: Is the "true score" a useful construct or mathematical fiction?
3. **Historical**: When did SEM acknowledgment decline in practice?
4. **Policy**: How do institutions justify ignoring measurement error?

## Expected Outputs

1. **Academic Paper**: "The Reification of Uncertainty: How IQ Testing Transforms Probability into Ontology"
2. **Data Package**: Curated dataset of psychometric critiques
3. **Interactive Tools**: Web-based demonstrations
4. **Policy Brief**: Recommendations for reporting practices

## Timeline

- **Month 1-2**: Literature collection and review
- **Month 3-4**: Statistical analysis and simulation
- **Month 5-6**: Writing and visualization
- **Month 7+**: Dissemination and iteration

---

*This is a living document. Update as research progresses.*
