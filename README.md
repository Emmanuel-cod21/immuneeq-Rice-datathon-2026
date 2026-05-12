ImmuneEQ — CD4/CD8 Trajectories, Treatment Heterogeneity & Equity-Informed Precision Medicine in HIV Care
Rice Datathon 2026 - Beginner Track

---

Overview
ImmuneEQ analyzes the AIDS Clinical Trials Group Study 175 dataset (1996) to investigate treatment heterogeneity and explore evidence for precision medicine approaches in HIV care. The analysis combines classical statistical methods with machine learning to reveal measurable differences in treatment response across demographic groups and transmission risk categories.

---

Key Findings

1. Treatment Effectiveness
- 55% of patients showed CD4 cell improvement over 20 weeks
- Mean CD4 increase: +30.2 cells/mm³ overall
- ZDV + ddI and ddI-only arms outperformed ZDV monotherapy

2. Machine Learning — Treatment Outcome Prediction
- Random Forest classifier**: Cross-validated AUC = 0.657
- Gradient Boosting classifier**: Cross-validated AUC = 0.651
- Top predictive features: baseline CD4, baseline CD8, body weight, age, prior antiretroviral therapy
- Race emerged as an independent predictive feature, validating the equity analysis

3. Model Equity Analysis
- By race: White AUC = 0.876, Non-white AUC = 0.878 — no algorithmic bias detected
- By gender: Female AUC = 0.920, Male AUC = 0.868 — model predicts female outcomes more accurately, suggesting differential treatment response patterns worth further investigation

4. Survival Analysis (Kaplan-Meier)
- Treatment arms show clear survival separation — combination therapy outperforms monotherapy
- Racial survival gap approaches but does not reach significance (log-rank p=0.0619), suggesting larger diverse trials are needed
- Transmission risk group survival: Hemophilia patients show notably worse outcomes vs other groups; IV drug use also associated with worse survival

5. Equity & Heterogeneity
- Measurable CD4 response difference: majority group +32.4 vs minority group +25.1 cells/mm³ (p < 0.05)
- Dataset limitation: race recorded as binary in 1996 trial protocol — modern studies use full ethnicity data

---

Methods

| Method | Purpose |
|--------|---------|
| Descriptive statistics | Baseline demographics, CD4/CD8 distributions |
| Independent t-tests | Group comparisons (α = 0.05) |
| Linear regression | Baseline outcome prediction (R² = 0.69) |
| Random Forest (n=200 trees) | Treatment outcome classification |
| Gradient Boosting (n=200, lr=0.05) | Treatment outcome classification |
| Kaplan-Meier + log-rank test | Survival analysis by subgroup |
| Subgroup AUC analysis | Model equity evaluation |

---

Technologies
- Python 3.x — pandas, numpy, matplotlib, seaborn
- scikit-learn — Random Forest, Gradient Boosting, cross-validation
- lifelines — Kaplan-Meier survival analysis
- ucimlrepo — dataset loading

---

Dataset
AIDS Clinical Trials Group Study 175
UC Irvine Machine Learning Repository  
DOI: https://doi.org/10.24432/C5JK5B  
2,139 patients · 4 treatment arms · 20-week follow-up · 1996

---

Running the Analysis
```bash
pip install ucimlrepo scikit-learn lifelines pandas numpy matplotlib seaborn
jupyter notebook aids_clinical_analysis.ipynb
```

---

Motivation
Modern HIV patients achieve viral suppression but face new challenges: metabolic side effects, performance capacity, and quality of life optimization. This project was motivated by a personal question — do different populations respond differently to HIV treatment? As someone living with HIV since infancy, the shift from survival-focused to performance-focused care is lived experience, not just a research framing.

> "In 1996, HIV patients fought to survive. In 2026, we optimize for performance."

---

Future Directions
- Survival interaction effects between demographics and treatment arms
- Integration with modern ART registry data
- Pharmacogenomic marker incorporation
- Personalized treatment recommendation system

---

Citation & Acknowledgments
- AIDS Clinical Trials Group Study 175 Dataset, UCI ML Repository
- Rice Datathon 2026
- Anthropic Claude — analysis assistance

License
MIT License# ImmuneEQ — CD4/CD8 Trajectories, Treatment Heterogeneity & Equity-Informed Precision Medicine in HIV Care
Rice Datathon 2026 - Beginner Track

---

Overview
ImmuneEQ analyzes the AIDS Clinical Trials Group Study 175 dataset (1996) to investigate treatment heterogeneity and explore evidence for precision medicine approaches in HIV care. The analysis combines classical statistical methods with machine learning to reveal measurable differences in treatment response across demographic groups and transmission risk categories.

---

Key Findings

1. Treatment Effectiveness
- 55% of patients showed CD4 cell improvement over 20 weeks
- Mean CD4 increase: +30.2 cells/mm³ overall
- ZDV + ddI and ddI-only arms outperformed ZDV monotherapy

2. Machine Learning — Treatment Outcome Prediction
- Random Forest classifier**: Cross-validated AUC = 0.657
- Gradient Boosting classifier**: Cross-validated AUC = 0.651
- Top predictive features: baseline CD4, baseline CD8, body weight, age, prior antiretroviral therapy
- Race emerged as an independent predictive feature, validating the equity analysis

### 3. Model Equity Analysis
- By race: White AUC = 0.876, Non-white AUC = 0.878 — no algorithmic bias detected
- By gender: Female AUC = 0.920, Male AUC = 0.868 — model predicts female outcomes more accurately, suggesting differential treatment response patterns worth further investigation

4. Survival Analysis (Kaplan-Meier)
- Treatment arms show clear survival separation — combination therapy outperforms monotherapy
- Racial survival gap approaches but does not reach significance (log-rank p=0.0619), suggesting larger diverse trials are needed
- Transmission risk group survival: Hemophilia patients show notably worse outcomes vs other groups; IV drug use also associated with worse survival

5. Equity & Heterogeneity
- Measurable CD4 response difference: majority group +32.4 vs minority group +25.1 cells/mm³ (p < 0.05)
- Dataset limitation: race recorded as binary in 1996 trial protocol — modern studies use full ethnicity data

---

Methods

| Method | Purpose |
|--------|---------|
| Descriptive statistics | Baseline demographics, CD4/CD8 distributions |
| Independent t-tests | Group comparisons (α = 0.05) |
| Linear regression | Baseline outcome prediction (R² = 0.69) |
| Random Forest (n=200 trees) | Treatment outcome classification |
| Gradient Boosting (n=200, lr=0.05) | Treatment outcome classification |
| Kaplan-Meier + log-rank test | Survival analysis by subgroup |
| Subgroup AUC analysis | Model equity evaluation |

---

Technologies
- Python 3.x — pandas, numpy, matplotlib, seaborn
- scikit-learn — Random Forest, Gradient Boosting, cross-validation
- lifelines — Kaplan-Meier survival analysis
- ucimlrepo — dataset loading

---

Dataset
AIDS Clinical Trials Group Study 175
UC Irvine Machine Learning Repository  
DOI: https://doi.org/10.24432/C5JK5B  
2,139 patients · 4 treatment arms · 20-week follow-up · 1996

---

Running the Analysis
```bash
pip install ucimlrepo scikit-learn lifelines pandas numpy matplotlib seaborn
jupyter notebook aids_clinical_analysis.ipynb
```

---

Motivation
Modern HIV patients achieve viral suppression but face new challenges: metabolic side effects, performance capacity, and quality of life optimization. This project was motivated by a personal question — do different populations respond differently to HIV treatment? As someone living with HIV since infancy, the shift from survival-focused to performance-focused care is lived experience, not just a research framing.

> "In 1996, HIV patients fought to survive. In 2026, we optimize for performance."

---

Future Directions
- Survival interaction effects between demographics and treatment arms
- Integration with modern ART registry data
- Pharmacogenomic marker incorporation
- Personalized treatment recommendation system

---

Citation & Acknowledgments
- AIDS Clinical Trials Group Study 175 Dataset, UCI ML Repository
- Rice Datathon 2026
- Anthropic Claude — analysis assistance

License
MIT License 
