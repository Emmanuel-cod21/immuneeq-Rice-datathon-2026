ImmuneEQ — CD4/CD8 Trajectories, Treatment Heterogeneity & Equity-Informed Precision Medicine in HIV Care
Rice Datathon 2026 - Beginner Track

Overview
This project analyzes the AIDS Clinical Trials Group Study 175 dataset (1996) to investigate treatment heterogeneity and explore evidence for precision medicine approaches in HIV care. The analysis reveals measurable differences in treatment response across demographic groups and demonstrates that incorporating demographic information improves outcome prediction.

Motivation
Modern HIV care has moved beyond survival. Patients today live normal lifespans but face new challenges: metabolic side effects, performance capacity, and quality of life optimization. This analysis was motivated by a personal question: do different populations respond differently to HIV treatment? Understanding treatment heterogeneity is essential for the next era of personalized HIV care.

Key Findings
1. Treatment Effectiveness (Overall)
71.2% of patients showed CD4 cell improvement
Mean CD4 increase: +30.2 cells/mm³
Immune function was successfully restored in most patients
2. Treatment Heterogeneity
Measurable differences in treatment response across demographic groups
Majority group: +32.4 cells/mm³ average CD4 change
Minority groups: +25.1 cells/mm³ average CD4 change
Difference of 7.3 cells/mm³ (statistically significant, p < 0.05)
3. Predictive Modeling Evidence
Model without demographic information: R² = 0.6847
Model with demographic information: R² = 0.6923
+1.11% improvement in predictive accuracy
Including demographic factors improves outcome prediction
4. Quality of Life
Mean Karnofsky score: 96.5/100
Most patients maintained functional capacity
52.3% improved CD4:CD8 ratio (key immune health indicator)
Clinical Context
HIV Immunology Basics
CD4 T-cells: Helper immune cells targeted by HIV (normal: 500-1500 cells/mm³)
CD8 T-cells: Cytotoxic immune cells (normal: 300-1000 cells/mm³)
CD4:CD8 Ratio: Critical immune health indicator (normal: >1.0)
AIDS Definition: CD4 count < 200 cells/mm³
Karnofsky Score: Functional capacity measure (0-100, where 100 = normal function)
Why This Matters
While this dataset is from 1996 (pre-HAART era), the core question remains highly relevant:

1996 Problem: Survival
2026 Problem: Performance, quality of life, metabolic health

Modern HIV patients achieve viral suppression (undetectable = untransmittable) but face new challenges:

Metabolic side effects (e.g., CK elevation during high-performance athletics)
Long-term cardiovascular health
Drug-drug interactions
Quality of life optimization
Understanding treatment heterogeneity enables precision medicine approaches that optimize outcomes for individual patients.

Dataset
Source: UC Irvine Machine Learning Repository

Citation:

AIDS Clinical Trials Group Study 175 Dataset
UC Irvine Machine Learning Repository
DOI: https://doi.org/10.24432/C5JK5B
Study Details:

Patients: 2,139
Time Period: 1996 (pre-HAART era)
Treatment Arms: 4 different antiretroviral combinations
Follow-up: 20 weeks
Endpoints: CD4/CD8 counts, survival, quality of life
Methodology
Analysis Pipeline
Data Loading & Exploration
Loaded dataset using ucimlrepo package
Examined 2,139 patients across 4 treatment arms
Assessed data quality and missingness
Demographic Analysis
Analyzed representation across gender, race, risk factors
Used modern equity framing (majority/minority groups)
Documented dataset limitations (1990s binary categorization)
Immune System Recovery Analysis
Calculated CD4:CD8 ratios at baseline and 20 weeks
Assessed proportion below normal threshold (<1.0)
Measured improvement rates
Treatment Effectiveness Analysis
Compared CD4/CD8 counts: baseline vs 20 weeks
Analyzed response by treatment arm
Calculated cell count changes
Equity & Heterogeneity Analysis
Examined outcomes across demographic groups
Performed statistical tests (t-tests) for significance
Analyzed treatment arm responses by demographic
Predictive Modeling
Built linear regression models to predict 20-week CD4 counts
Model 1: Baseline CD4 + Treatment Arm
Model 2: Baseline CD4 + Treatment Arm + Demographics
Compared R² scores and RMSE
Quality of Life Analysis
Analyzed Karnofsky performance status scores
Examined relationship with prior antiretroviral therapy
Compared scores across demographic groups
Statistical Methods
Descriptive Statistics: Means, medians, percentages
Inferential Statistics: Independent t-tests for group comparisons
Regression Analysis: Linear regression with R², RMSE metrics
Significance Level: α = 0.05
Technologies Used
Python 3.x
pandas - Data manipulation
numpy - Numerical computations
matplotlib & seaborn - Visualizations
scikit-learn - Statistical modeling
scipy - Statistical tests

Karnofsky performance status analysis
Functional capacity outcomes
Running the Code
Prerequisites
bash
pip install ucimlrepo pandas numpy matplotlib seaborn scikit-learn scipy
Execution
Clone the repository:
bash
git clone https://github.com/[your-username]/aids-precision-medicine
cd aids-precision-medicine
Run the analysis:
bash
jupyter notebook aids_clinical_analysis.ipynb
Or run as Python script:

bash
python aids_analysis_part1.py
python aids_analysis_part2.py
Expected Output
6 PNG visualization files (saved automatically)
Console output with statistical summaries
Model performance metrics
Project Structure
aids-precision-medicine/
│
├── aids_clinical_analysis.ipynb    # Main analysis notebook
├── aids_analysis_part1.py           # Part 1: Demographics, CD4/CD8, Treatment
├── aids_analysis_part2.py           # Part 2: Statistical models, Summary
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
│
├── visualizations/
│   ├── 01_demographics.png
│   ├── 02_cd4_cd8_ratios.png
│   ├── 03_treatment_effectiveness.png
│   ├── 04_equity_analysis.png
│   ├── 05_statistical_models.png
│   └── 06_quality_of_life.png
│
└── presentation/
    ├── slides.pdf
    └── script.md
Social Impact
Health Equity Implications
This analysis demonstrates the importance of:

Diverse Clinical Trial Representation
Historically underrepresented groups must be included
Treatment effectiveness may vary across populations
One-size-fits-all approaches may not optimize outcomes for everyone
Precision Medicine in HIV Care
Future treatment should be personalized
Demographic, genetic, and physiological factors matter
Performance metrics beyond viral suppression are important
Modern HIV Care Challenges
Survival → Performance & Quality of Life
Metabolic phenotyping (e.g., CK levels during athletics)
Long-term health optimization
Personal Context
As someone living with HIV since infancy and experiencing modern antiretroviral therapy, this project was motivated by a lived question: why do different people respond differently to treatment? During high-intensity military physical conditioning, CK elevation revealed that modern HIV care extends beyond immune recovery to include performance physiology - a dimension not captured in historical clinical trials.

Limitations
Historical Dataset (1996)
Pre-HAART era treatments (less effective than modern ART)
Binary racial categorization (limitation of 1990s data collection)
Limited demographic variables
Statistical Constraints
Linear models may not capture non-linear relationships
Potential confounding variables not controlled
Limited sample size for subgroup analyses
Generalizability
Results specific to tested treatment regimens
May not apply to modern combination therapies
Dataset lacks some modern biomarkers
Future Directions
Extended Analysis
Survival outcome analysis
Interaction effects between demographics and treatment arms
Time-series modeling of immune recovery
Modern Data Integration
Compare 1990s outcomes to contemporary registry data
Incorporate pharmacogenomic markers
Include metabolic and performance biomarkers
Precision Medicine Development
Build personalized treatment recommendation systems
Integrate genetic, demographic, and physiological data
Develop performance-optimized treatment protocols
References
AIDS Clinical Trials Group Study 175 Dataset. UC Irvine Machine Learning Repository. DOI: https://doi.org/10.24432/C5JK5B
CD4-CD8 Ratio. University of Rochester Medical Center, 2025. https://www.urmc.rochester.edu/encyclopedia/content?contenttypeid=167&contentid=cd4_cd8_ratio
Karnofsky Performance Status Scale. Medscape, Christensen, Buck, January 23, 2025. https://emedicine.medscape.com/article/2172510-overview
Hammer SM, et al. "A trial comparing nucleoside monotherapy with combination therapy in HIV-infected adults with CD4 cell counts from 200 to 500 per cubic millimeter." New England Journal of Medicine, 1996.
Acknowledgments
Rice Datathon 2026 for providing the platform
UC Irvine ML Repository for dataset access
AIDS Clinical Trials Group for the original research
Anthropic Claude for analysis assistance
License
This project is licensed under the MIT License - see LICENSE file for details.

Contact
For questions about this analysis:

GitHub Issues: Open an issue
Email: Uomaemmanuel@myyahoo.com
Final Thought
"Precision medicine will define the next era of HIV care."

In 1996, HIV patients fought to survive.
In 2026, we optimize for performance.
The data shows us why personalized approaches matter.

This project was completed for Rice Datathon 2026 - Beginner Track

