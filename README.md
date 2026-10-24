# ImmuneEQ: CD4 Response, Treatment Heterogeneity, and Subgroup Model Performance in HIV Care
**Rice Datathon 2026, Beginner Track**

Live dashboard: [immuneeq.streamlit.app](https://immuneeq.streamlit.app)

---

## Overview
ImmuneEQ analyzes the AIDS Clinical Trials Group Study 175 (ACTG 175) dataset (1996) to ask whether CD4 response to HIV treatment and the performance of a simple outcome-prediction model differ across demographic groups. The project combines descriptive statistics, tree-based classifiers, and Kaplan-Meier survival analysis. Results below are limited to what has been checked against the data; weaker or inconclusive findings are reported as such.

---

## Results

### 1. Treatment outcomes
- 2,139 patients across 4 treatment arms. 55.0% (1,176) had any increase in CD4 at 20 weeks.
- Mean CD4 rose from 350.5 to 371.3 cells/mm³ (mean change +20.8).
- "Improved" means CD4 at 20 weeks is higher than baseline by any amount, so it includes small, noisy changes.

### 2. Raw CD4 change by race (binary variable recorded in 1996)
- White: +24.0 cells/mm³ (n = 1,522). Non-white: +12.9 cells/mm³ (n = 617). Difference: 11.1.
- Welch t-test: t = 1.96, p = 0.0506, 95% CI for the difference [-0.0, 22.3].
- This is suggestive but **not statistically significant at the 0.05 level**, and it is unadjusted for baseline CD4 or treatment arm.

### 3. Outcome prediction (Random Forest, Gradient Boosting)
Target: any CD4 increase at 20 weeks.

| Model | 5-fold CV ROC-AUC | Held-out test ROC-AUC |
|-------|-------------------|-----------------------|
| Random Forest (200 trees, depth 8) | 0.657 | 0.659 |
| Gradient Boosting (200 trees, lr 0.05) | 0.651 | not reported |

- AUC near 0.66 is modest: better than chance, far from clinically useful.
- Most important features in the Random Forest: baseline CD4, baseline CD8, body weight, age, prior antiretroviral therapy, treatment arm.

### 4. Subgroup model performance (held-out test set, n = 428)
AUC is computed only on patients the Random Forest did not train on.

| Subgroup | n (test) | Held-out AUC |
|----------|----------|--------------|
| White | 307 | 0.682 |
| Non-white | 121 | 0.591 |
| Female | 73 | 0.676 |
| Male | 355 | 0.656 |

- Discrimination is weakest for the non-white group, but that group has only 121 test patients, so the estimate is noisy. The dashboard reports 95% bootstrap confidence intervals for each subgroup.
- The data cannot establish either the presence or absence of model bias.

### 5. Survival analysis (Kaplan-Meier)
- Race (binary): log-rank p = 0.0619, not significant.
- Treatment arm and transmission risk group: curves are shown in the notebook and dashboard. The dashboard computes log-rank p-values live for the selected patients. Group differences should not be read from the curves without those p-values. Curves are unadjusted, and some groups are small (hemophilia n = 180, IV drug use n = 276).

---

## Methods

| Method | Purpose |
|--------|---------|
| Welch t-test | CD4 change by race |
| Random Forest, Gradient Boosting | Predict any CD4 increase |
| Stratified 5-fold CV and 80/20 held-out split (seed 42) | Estimate ROC-AUC |
| Held-out subgroup AUC with bootstrap CIs | Compare model performance across groups |
| Kaplan-Meier, log-rank tests | Survival by treatment arm, race, risk group |

---

## Limitations
- **Race is binary.** The 1996 protocol recorded race as white vs non-white. This is a dataset limitation, not a modeling choice, and it hides most of the variation that matters.
- **Small subgroups.** Subgroup estimates (especially non-white and female test sets) have wide uncertainty.
- **Weak outcome definition.** Any CD4 increase, however small, is a noisy label.
- **Unadjusted comparisons.** Group differences are not adjusted for baseline CD4, treatment arm, or other covariates.
- **Old data.** Treatments in this trial (ZDV, ddI, zalcitabine) are not current standard of care.

---

## Corrections to earlier versions
An earlier version of this README reported a mean CD4 increase of +30.2, a 7.3 cells/mm³ significant race gap, and "no algorithmic bias detected" based on subgroup AUCs of about 0.88 computed on all patients, including training data. Re-checking showed the first two did not reproduce and the third was an in-sample artifact. The numbers above replace them.

---

## Technologies
Python, pandas, numpy, scipy, scikit-learn, lifelines, matplotlib, Streamlit, ucimlrepo.

## Dataset
AIDS Clinical Trials Group Study 175, UC Irvine Machine Learning Repository.
DOI: https://doi.org/10.24432/C5JK5B
2,139 patients, 4 treatment arms, 1996.

## Running the analysis
```bash
pip install -r requirements.txt
jupyter notebook aids_clinical_analysis.ipynb
streamlit run app.py
```

---

## Motivation
Modern HIV patients achieve viral suppression but face new challenges: metabolic side effects, performance capacity, and quality of life optimization. This project grew out of a personal question: do different populations respond differently to HIV treatment? As someone living with HIV since infancy, the shift from survival-focused to performance-focused care is lived experience, not just a research framing.

> *"In 1996, HIV patients fought to survive. In 2026, we optimize for performance."*

## Future directions
- Adjusted models (baseline CD4, treatment arm) for group differences in CD4 response
- A less noisy outcome definition, such as a clinically meaningful CD4 increase
- Modern ART registry data with richer race and ethnicity coding
- Pharmacogenomic markers

## Acknowledgments
- AIDS Clinical Trials Group Study 175 dataset, UCI ML Repository
- Rice Datathon 2026

## License
MIT License
