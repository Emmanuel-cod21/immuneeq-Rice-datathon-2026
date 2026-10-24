import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from ucimlrepo import fetch_ucirepo
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="ImmuneEQ", page_icon="🧬", layout="wide")

st.title("🧬 ImmuneEQ")
st.markdown("**CD4/CD8 Trajectories · Treatment Heterogeneity · Equity-Informed Precision Medicine in HIV Care**")
st.caption("Rice Datathon 2026 · AIDS Clinical Trials Group Study 175 · 2,139 patients")

FEATURE_COLS = ['cd40', 'cd80', 'trt', 'age', 'wtkg', 'karnof',
                'preanti', 'race', 'gender', 'homo', 'drugs',
                'oprior', 'z30', 'zprior', 'str2', 'strat', 'symptom']
TRT_LABELS = {0: 'ZDV only', 1: 'ZDV + ddI', 2: 'ZDV + Zal', 3: 'ddI only'}


# ── Helpers ───────────────────────────────────────────────────────────────────
def bootstrap_auc_ci(y_true, y_score, n_boot=1000, seed=42):
    """95% percentile bootstrap CI for ROC-AUC."""
    rng = np.random.default_rng(seed)
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return lo, hi


# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    aids = fetch_ucirepo(id=890)
    df = aids.data.features.copy()
    df['cid'] = aids.data.targets.values
    df['cd4_change'] = df['cd420'] - df['cd40']
    df['cd4_improved'] = (df['cd4_change'] > 0).astype(int)

    def get_risk_group(row):
        if row['hemo'] == 1:
            return 'Hemophilia'
        elif row['drugs'] == 1:
            return 'IV Drug Use'
        elif row['homo'] == 1:
            return 'Homosexual Contact'
        return 'Heterosexual/Other'

    df['risk_group'] = df.apply(get_risk_group, axis=1)
    return df


@st.cache_resource
def train_model(df):
    X, y = df[FEATURE_COLS], df['cd4_improved']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    gb.fit(X_train, y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = {
        'Random Forest': cross_val_score(rf, X, y, cv=cv, scoring='roc_auc').mean(),
        'Gradient Boosting': cross_val_score(gb, X, y, cv=cv, scoring='roc_auc').mean(),
    }

    # Held-out subgroup performance (test set only, never in-sample)
    test_df = df.loc[X_test.index]
    test_scores = rf.predict_proba(X_test)[:, 1]
    rows = []
    for col, val, label in [('race', 0, 'White'), ('race', 1, 'Non-white'),
                            ('gender', 0, 'Female'), ('gender', 1, 'Male')]:
        m = (test_df[col] == val).values
        auc = roc_auc_score(y_test[m], test_scores[m])
        lo, hi = bootstrap_auc_ci(y_test[m], test_scores[m])
        rows.append({'Group': label, 'Variable': col, 'n (test set)': int(m.sum()),
                     'Held-out AUC': auc, 'CI low': lo, 'CI high': hi})
    subgroup = pd.DataFrame(rows)
    pooled_auc = roc_auc_score(y_test, test_scores)
    return rf, gb, X_test, y_test, cv_auc, subgroup, pooled_auc


with st.spinner("Loading dataset and training models (first load takes about a minute)..."):
    df = load_data()
    rf, gb, X_test, y_test, cv_auc, subgroup, pooled_auc = train_model(df)

# ── Sidebar Filters ───────────────────────────────────────────────────────────
st.sidebar.header("🔬 Filter Patients")

trt_options = st.sidebar.multiselect("Treatment Arm", options=list(TRT_LABELS.values()),
                                     default=list(TRT_LABELS.values()))

st.sidebar.markdown("**Race**")
st.sidebar.caption("⚠️ The 1996 trial recorded race as binary. This is a dataset limitation, not our classification.")
race_options = st.sidebar.multiselect("Race", options=['White', 'Non-white*'],
                                      default=['White', 'Non-white*'],
                                      label_visibility="collapsed")

gender_options = st.sidebar.multiselect("Gender", options=['Male', 'Female'], default=['Male', 'Female'])
risk_options = st.sidebar.multiselect("Transmission Risk Group",
                                      options=df['risk_group'].unique().tolist(),
                                      default=df['risk_group'].unique().tolist())

trt_vals = [k for k, v in TRT_LABELS.items() if v in trt_options]
race_vals = []
if 'White' in race_options:
    race_vals.append(0)
if 'Non-white*' in race_options:
    race_vals.append(1)
gender_vals = [1 if g == 'Male' else 0 for g in gender_options]

filtered = df[
    df['trt'].isin(trt_vals) &
    df['race'].isin(race_vals) &
    df['gender'].isin(gender_vals) &
    df['risk_group'].isin(risk_options)
]

st.sidebar.markdown(f"**Patients selected: {len(filtered):,}**")

if len(filtered) == 0:
    st.warning("No patients match the current filters. Widen the selection in the sidebar.")
    st.stop()

# ── Key Metrics ───────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns(2)
col1.metric("Patients Selected", f"{len(filtered):,}")
col2.metric("Any CD4 Increase at 20 Weeks", f"{filtered['cd4_improved'].mean():.1%}")
col3, col4 = st.columns(2)
col3.metric("Baseline CD4 (cells/mm³)", f"{filtered['cd40'].mean():.0f}")
col4.metric("CD4 at 20 Weeks (cells/mm³)", f"{filtered['cd420'].mean():.0f}")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Treatment Outcomes", "🤖 ML Prediction",
                                  "📈 Survival Analysis", "⚖️ Equity Analysis"])

# ── Tab 1: Treatment Outcomes ─────────────────────────────────────────────────
with tab1:
    st.subheader("CD4 Change by Treatment Arm")
    st.caption("'Improved' means CD4 at 20 weeks is higher than baseline by any amount, so it includes small, "
               "noisy changes.")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

    improve_by_trt = filtered.groupby('trt')['cd4_improved'].mean()
    axes[0].bar([TRT_LABELS[t] for t in improve_by_trt.index], improve_by_trt.values,
                color=colors[:len(improve_by_trt)])
    axes[0].set_title('Proportion With Any CD4 Increase, by Treatment Arm')
    axes[0].set_ylabel('Proportion Improved')
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis='y', alpha=0.3)

    cd4_change = filtered['cd4_change']
    axes[1].hist(cd4_change, bins=40, color='#2196F3', alpha=0.7, edgecolor='white')
    axes[1].axvline(0, color='red', linestyle='--', label='No change')
    axes[1].axvline(cd4_change.mean(), color='green', linestyle='--', label=f'Mean: {cd4_change.mean():.1f}')
    axes[1].set_title('Distribution of CD4 Change (Baseline to 20 weeks)')
    axes[1].set_xlabel('CD4 Change (cells/mm³)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Tab 2: ML Prediction ──────────────────────────────────────────────────────
with tab2:
    st.subheader("Treatment Outcome Prediction (Random Forest and Gradient Boosting)")
    m1, m2, m3 = st.columns(3)
    m1.metric("Random Forest, 5-fold CV AUC", f"{cv_auc['Random Forest']:.3f}")
    m2.metric("Gradient Boosting, 5-fold CV AUC", f"{cv_auc['Gradient Boosting']:.3f}")
    m3.metric("Random Forest, held-out test AUC", f"{pooled_auc:.3f}")
    st.caption("AUC around 0.66 is modest. The models separate outcomes better than chance but are far from "
               "clinically usable. Baseline CD4 and CD8 carry most of the signal.")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(7, 5))
        for model, name, color in [(rf, 'Random Forest', '#2196F3'), (gb, 'Gradient Boosting', '#4CAF50')]:
            scores = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, scores)
            ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (test AUC={roc_auc_score(y_test, scores):.3f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Chance (AUC=0.500)')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves (held-out test set)')
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(7, 5))
        importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True).tail(10)
        importances.plot(kind='barh', ax=ax, color='#2196F3', alpha=0.8)
        ax.set_title('Top 10 Feature Importances\n(Random Forest)')
        ax.set_xlabel('Importance Score')
        ax.grid(axis='x', alpha=0.3)
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("🔮 Explore a Hypothetical Patient")
    st.caption("Research demo only. This is not a clinical tool and should not inform any treatment decision.")
    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        p_cd40 = st.slider("Baseline CD4 (cells/mm³)", 0, 1000, 350)
        p_cd80 = st.slider("Baseline CD8 (cells/mm³)", 0, 2000, 700)
        p_age = st.slider("Age", 18, 70, 35)
        p_wtkg = st.slider("Weight (kg)", 40, 120, 75)
        p_karnof = st.slider("Karnofsky Score", 50, 100, 96)
        p_preanti = st.slider("Prior ART (months)", 0, 60, 12)
    with p_col2:
        p_trt = st.selectbox("Treatment Arm", options=[0, 1, 2, 3], format_func=lambda x: TRT_LABELS[x])
        p_race = st.selectbox("Race (1996 binary)", options=[0, 1], format_func=lambda x: 'White' if x == 0 else 'Non-white')
        p_gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
        p_homo = st.selectbox("Homosexual Contact", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
        p_drugs = st.selectbox("IV Drug Use", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    with p_col3:
        p_oprior = st.selectbox("Non-ZDV ART Prior", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
        p_z30 = st.selectbox("ZDV in Prior 30 Days", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
        p_zprior = st.selectbox("ZDV Prior to Trial", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
        p_str2 = st.selectbox("ART History (str2)", options=[0, 1])
        p_strat = st.selectbox("ART History (strat)", options=[1, 2, 3])
        p_symptom = st.selectbox("Symptomatic", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')

    patient = pd.DataFrame([[p_cd40, p_cd80, p_trt, p_age, p_wtkg, p_karnof,
                             p_preanti, p_race, p_gender, p_homo, p_drugs,
                             p_oprior, p_z30, p_zprior, p_str2, p_strat, p_symptom]],
                           columns=FEATURE_COLS)
    prob = rf.predict_proba(patient)[0][1]
    st.info(f"Model's estimated probability of any CD4 increase: **{prob:.1%}** "
            f"(model AUC is only about 0.66, so treat this as illustrative)")

# ── Tab 3: Survival Analysis ──────────────────────────────────────────────────
with tab3:
    st.subheader("Kaplan-Meier Survival Analysis")
    st.caption("Curves are unadjusted. p-values come from log-rank tests on the currently filtered patients. "
               "Small groups give wide uncertainty, so do not read differences into curves without a p-value.")

    def fmt_p(p):
        return "n/a" if p is None else f"p={p:.4f}"

    # Treatment arm test
    trt_groups = filtered[filtered['trt'].isin([k for k in TRT_LABELS if (filtered['trt'] == k).sum() >= 5])]
    p_trt_lr = None
    if trt_groups['trt'].nunique() >= 2:
        p_trt_lr = multivariate_logrank_test(trt_groups['time'], trt_groups['trt'], trt_groups['cid']).p_value

    # Race test
    p_race_lr = None
    wmask, nmask = filtered['race'] == 0, filtered['race'] == 1
    if wmask.sum() >= 5 and nmask.sum() >= 5:
        p_race_lr = logrank_test(filtered.loc[wmask, 'time'], filtered.loc[nmask, 'time'],
                                 event_observed_A=filtered.loc[wmask, 'cid'],
                                 event_observed_B=filtered.loc[nmask, 'cid']).p_value

    # Risk group test
    rg_counts = filtered['risk_group'].value_counts()
    rg_keep = rg_counts[rg_counts >= 5].index
    rg_df = filtered[filtered['risk_group'].isin(rg_keep)]
    p_rg_lr = None
    if rg_df['risk_group'].nunique() >= 2:
        p_rg_lr = multivariate_logrank_test(rg_df['time'], rg_df['risk_group'], rg_df['cid']).p_value

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

    for trt_val, color in zip([0, 1, 2, 3], colors):
        mask = filtered['trt'] == trt_val
        if mask.sum() < 5:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(filtered.loc[mask, 'time'], event_observed=filtered.loc[mask, 'cid'], label=TRT_LABELS[trt_val])
        kmf.plot_survival_function(ax=axes[0], color=color, ci_show=True)
    axes[0].set_title(f'By Treatment Arm\n(log-rank {fmt_p(p_trt_lr)})')

    for race_val, label, color in [(0, 'White', '#2196F3'), (1, 'Non-white', '#FF9800')]:
        mask = filtered['race'] == race_val
        if mask.sum() < 5:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(filtered.loc[mask, 'time'], event_observed=filtered.loc[mask, 'cid'], label=label)
        kmf.plot_survival_function(ax=axes[1], color=color, ci_show=True)
    axes[1].set_title(f'By Race (1996 binary classification)\n(log-rank {fmt_p(p_race_lr)})')

    risk_colors = {'Homosexual Contact': '#2196F3', 'IV Drug Use': '#E91E63',
                   'Hemophilia': '#4CAF50', 'Heterosexual/Other': '#FF9800'}
    for group, color in risk_colors.items():
        mask = filtered['risk_group'] == group
        if mask.sum() < 5:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(filtered.loc[mask, 'time'], event_observed=filtered.loc[mask, 'cid'],
                label=f'{group} (n={mask.sum()})')
        kmf.plot_survival_function(ax=axes[2], color=color, ci_show=False)
    axes[2].set_title(f'By Transmission Risk Group\n(log-rank {fmt_p(p_rg_lr)})')

    for ax in axes:
        ax.set_xlabel('Days')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('Survival Probability')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Tab 4: Equity Analysis ────────────────────────────────────────────────────
with tab4:
    st.subheader("Equity Analysis")
    st.caption("⚠️ Race was recorded as binary in the 1996 trial protocol. This is a dataset limitation, "
               "not our classification choice.")

    st.markdown("#### 1. Raw CD4 change by race (all 2,139 patients)")
    w = df.loc[df['race'] == 0, 'cd4_change']
    nw = df.loc[df['race'] == 1, 'cd4_change']
    diff = w.mean() - nw.mean()
    se = np.sqrt(w.var(ddof=1) / len(w) + nw.var(ddof=1) / len(nw))
    t_stat, p_val = stats.ttest_ind(w, nw, equal_var=False)
    c1, c2, c3 = st.columns(3)
    c1.metric(f"White (n={len(w):,})", f"{w.mean():+.1f} cells/mm³")
    c2.metric(f"Non-white (n={len(nw):,})", f"{nw.mean():+.1f} cells/mm³")
    c3.metric("Difference", f"{diff:.1f} cells/mm³")
    st.write(f"Welch t-test: t = {t_stat:.2f}, p = {p_val:.4f}, "
             f"95% CI for the difference [{diff - 1.96 * se:.1f}, {diff + 1.96 * se:.1f}].")
    if p_val >= 0.05:
        st.info("This difference is suggestive but not statistically significant at the 0.05 level.")
    else:
        st.info("This difference is statistically significant at the 0.05 level. It is unadjusted for "
                "baseline CD4, treatment arm, or other covariates.")

    st.markdown("#### 2. Model performance by subgroup (held-out test set)")
    st.caption("AUC is computed only on patients the model did not train on, with 95% bootstrap confidence "
               "intervals. Subgroups are small, so intervals are wide.")
    display = subgroup.copy()
    display['95% CI'] = display.apply(lambda r: f"[{r['CI low']:.3f}, {r['CI high']:.3f}]", axis=1)
    display['Held-out AUC'] = display['Held-out AUC'].map(lambda x: f"{x:.3f}")
    st.dataframe(display[['Variable', 'Group', 'n (test set)', 'Held-out AUC', '95% CI']],
                 hide_index=True, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    y_pos = np.arange(len(subgroup))[::-1]
    ax.errorbar(subgroup['Held-out AUC'], y_pos,
                xerr=[subgroup['Held-out AUC'] - subgroup['CI low'], subgroup['CI high'] - subgroup['Held-out AUC']],
                fmt='o', color='#2196F3', capsize=4)
    ax.axvline(0.5, color='k', linestyle='--', lw=1, label='Chance')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{g} (n={n})" for g, n in zip(subgroup['Group'], subgroup['n (test set)'])])
    ax.set_xlabel('Held-out AUC (95% bootstrap CI)')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.warning("Interpretation: point estimates differ between groups, but the confidence intervals overlap and "
               "the test-set subgroups are small (the non-white and female groups especially). This analysis "
               "cannot establish either the presence or the absence of model bias.")

st.markdown("---")
st.caption("ImmuneEQ · Rice Datathon 2026 · Emmanuel Uzoma · Data: AIDS Clinical Trials Group Study 175, UCI ML Repository")
