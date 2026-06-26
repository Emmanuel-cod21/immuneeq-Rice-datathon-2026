import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="ImmuneEQ", page_icon="🧬", layout="wide")

st.title("🧬 ImmuneEQ")
st.markdown("**CD4/CD8 Trajectories · Treatment Heterogeneity · Equity-Informed Precision Medicine in HIV Care**")
st.caption("Rice Datathon 2026 · AIDS Clinical Trials Group Study 175 · 2,139 patients")

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    aids = fetch_ucirepo(id=890)
    df = aids.data.features.copy()
    df['cid'] = aids.data.targets.values
    df['cd4_improved'] = (df['cd420'] > df['cd40']).astype(int)
    def get_risk_group(row):
        if row['hemo'] == 1: return 'Hemophilia'
        elif row['drugs'] == 1: return 'IV Drug Use'
        elif row['homo'] == 1: return 'Homosexual Contact'
        else: return 'Heterosexual/Other'
    df['risk_group'] = df.apply(get_risk_group, axis=1)
    return df

@st.cache_resource
def train_model(df):
    feature_cols = ['cd40', 'cd80', 'trt', 'age', 'wtkg', 'karnof',
                    'preanti', 'race', 'gender', 'homo', 'drugs',
                    'oprior', 'z30', 'zprior', 'str2', 'strat', 'symptom']
    X = df[feature_cols]
    y = df['cd4_improved']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    gb.fit(X_train, y_train)
    return rf, gb, feature_cols, X_test, y_test

with st.spinner("Loading dataset and training model..."):
    df = load_data()
    rf, gb, feature_cols, X_test, y_test = train_model(df)

# ── Sidebar Filters ───────────────────────────────────────────────────────────
st.sidebar.header("🔬 Filter Patients")

trt_map = {0: 'ZDV only', 1: 'ZDV + ddI', 2: 'ZDV + Zal', 3: 'ddI only'}
trt_options = st.sidebar.multiselect("Treatment Arm", options=list(trt_map.values()), default=list(trt_map.values()))

st.sidebar.markdown("**Race**")
st.sidebar.caption("⚠️ 1996 dataset recorded race as binary — not a reflection of our classification.")
race_options = st.sidebar.multiselect("", options=['White', 'Non-white*'], 
                                       default=['White', 'Non-white*'],
                                       label_visibility="collapsed")

gender_options = st.sidebar.multiselect("Gender", options=['Male', 'Female'], default=['Male', 'Female'])
risk_options = st.sidebar.multiselect("Transmission Risk Group", 
                                       options=df['risk_group'].unique().tolist(), 
                                       default=df['risk_group'].unique().tolist())

trt_vals = [k for k, v in trt_map.items() if v in trt_options]
race_vals = []
if 'White' in race_options: race_vals.append(0)
if 'Non-white*' in race_options: race_vals.append(1)
gender_vals = [1 if g == 'Male' else 0 for g in gender_options]

filtered = df[
    df['trt'].isin(trt_vals) &
    df['race'].isin(race_vals) &
    df['gender'].isin(gender_vals) &
    df['risk_group'].isin(risk_options)
]

st.sidebar.markdown(f"**Patients selected: {len(filtered):,}**")

# ── Key Metrics ───────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns(2)
col1.metric("Patients Selected", f"{len(filtered):,}")
col2.metric("CD4 Improvement Rate", f"{filtered['cd4_improved'].mean():.1%}")
col3, col4 = st.columns(2)
col3.metric("Baseline CD4 (cells/mm³)", f"{filtered['cd40'].mean():.0f}")
col4.metric("CD4 at 20 Weeks (cells/mm³)", f"{filtered['cd420'].mean():.0f}")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Treatment Outcomes", "🤖 ML Prediction", "📈 Survival Analysis", "⚖️ Equity Analysis"])

# ── Tab 1: Treatment Outcomes ─────────────────────────────────────────────────
with tab1:
    st.subheader("CD4 Change by Treatment Arm")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    trt_labels = {0: 'ZDV only', 1: 'ZDV + ddI', 2: 'ZDV + Zal', 3: 'ddI only'}
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

    improve_by_trt = filtered.groupby('trt')['cd4_improved'].mean()
    axes[0].bar([trt_labels[t] for t in improve_by_trt.index], improve_by_trt.values, color=colors[:len(improve_by_trt)])
    axes[0].set_title('CD4 Improvement Rate by Treatment Arm')
    axes[0].set_ylabel('Proportion Improved')
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis='y', alpha=0.3)

    cd4_change = filtered['cd420'] - filtered['cd40']
    axes[1].hist(cd4_change, bins=40, color='#2196F3', alpha=0.7, edgecolor='white')
    axes[1].axvline(0, color='red', linestyle='--', label='No change')
    axes[1].axvline(cd4_change.mean(), color='green', linestyle='--', label=f'Mean: {cd4_change.mean():.1f}')
    axes[1].set_title('Distribution of CD4 Change (Baseline → 20 weeks)')
    axes[1].set_xlabel('CD4 Change (cells/mm³)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Tab 2: ML Prediction ──────────────────────────────────────────────────────
with tab2:
    st.subheader("Random Forest — Treatment Outcome Prediction")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(7, 5))
        for model, name, color in [(rf, 'Random Forest', '#2196F3'), (gb, 'Gradient Boost', '#4CAF50')]:
            fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={auc:.3f})')
        ax.plot([0,1],[0,1],'k--', lw=1, label='Random (AUC=0.500)')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(7, 5))
        importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True).tail(10)
        importances.plot(kind='barh', ax=ax, color='#2196F3', alpha=0.8)
        ax.set_title('Top 10 Feature Importances\n(Random Forest)')
        ax.set_xlabel('Importance Score')
        ax.grid(axis='x', alpha=0.3)
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("🔮 Predict Individual Patient Outcome")
    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        p_cd40 = st.slider("Baseline CD4 (cells/mm³)", 0, 1000, 350)
        p_cd80 = st.slider("Baseline CD8 (cells/mm³)", 0, 2000, 700)
        p_age = st.slider("Age", 18, 70, 35)
        p_wtkg = st.slider("Weight (kg)", 40, 120, 75)
        p_karnof = st.slider("Karnofsky Score", 50, 100, 96)
        p_preanti = st.slider("Prior ART (months)", 0, 60, 12)
    with p_col2:
        p_trt = st.selectbox("Treatment Arm", options=[0,1,2,3], format_func=lambda x: trt_labels[x])
        p_race = st.selectbox("Race", options=[0,1], format_func=lambda x: 'White' if x==0 else 'Non-white')
        p_gender = st.selectbox("Gender", options=[0,1], format_func=lambda x: 'Female' if x==0 else 'Male')
        p_homo = st.selectbox("Homosexual Contact", options=[0,1], format_func=lambda x: 'No' if x==0 else 'Yes')
        p_drugs = st.selectbox("IV Drug Use", options=[0,1], format_func=lambda x: 'No' if x==0 else 'Yes')
    with p_col3:
        p_oprior = st.selectbox("Non-ZDV ART Prior", options=[0,1], format_func=lambda x: 'No' if x==0 else 'Yes')
        p_z30 = st.selectbox("ZDV in Prior 30 Days", options=[0,1], format_func=lambda x: 'No' if x==0 else 'Yes')
        p_zprior = st.selectbox("ZDV Prior to Trial", options=[0,1], format_func=lambda x: 'No' if x==0 else 'Yes')
        p_str2 = st.selectbox("ART History (str2)", options=[0,1])
        p_strat = st.selectbox("ART History (strat)", options=[1,2,3])
        p_symptom = st.selectbox("Symptomatic", options=[0,1], format_func=lambda x: 'No' if x==0 else 'Yes')

    patient = np.array([[p_cd40, p_cd80, p_trt, p_age, p_wtkg, p_karnof,
                         p_preanti, p_race, p_gender, p_homo, p_drugs,
                         p_oprior, p_z30, p_zprior, p_str2, p_strat, p_symptom]])
    prob = rf.predict_proba(patient)[0][1]

    st.markdown("---")
    if prob >= 0.6:
        st.success(f"✅ Predicted probability of CD4 improvement: **{prob:.1%}**")
    elif prob >= 0.4:
        st.warning(f"⚠️ Predicted probability of CD4 improvement: **{prob:.1%}**")
    else:
        st.error(f"❌ Predicted probability of CD4 improvement: **{prob:.1%}**")

# ── Tab 3: Survival Analysis ──────────────────────────────────────────────────
with tab3:
    st.subheader("Kaplan-Meier Survival Analysis")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
    for trt_val, color in zip([0,1,2,3], colors):
        mask = filtered['trt'] == trt_val
        if mask.sum() < 5: continue
        kmf = KaplanMeierFitter()
        kmf.fit(filtered.loc[mask, 'time'], event_observed=filtered.loc[mask, 'cid'], label=trt_labels[trt_val])
        kmf.plot_survival_function(ax=axes[0], color=color, ci_show=True)
    axes[0].set_title('Survival by Treatment Arm')
    axes[0].set_xlabel('Days')
    axes[0].set_ylabel('Survival Probability')
    axes[0].grid(alpha=0.3)

    for race_val, label, color in [(0, 'White', '#2196F3'), (1, 'Non-white', '#FF9800')]:
        mask = filtered['race'] == race_val
        if mask.sum() < 5: continue
        kmf = KaplanMeierFitter()
        kmf.fit(filtered.loc[mask, 'time'], event_observed=filtered.loc[mask, 'cid'], label=label)
        kmf.plot_survival_function(ax=axes[1], color=color, ci_show=True)
    axes[1].set_title('Survival by Race\n(⚠ Binary classification: 1996 dataset limitation)')
    axes[1].set_xlabel('Days')
    axes[1].grid(alpha=0.3)

    risk_colors = {'Homosexual Contact': '#2196F3', 'IV Drug Use': '#E91E63',
                   'Hemophilia': '#4CAF50', 'Heterosexual/Other': '#FF9800'}
    for group, color in risk_colors.items():
        mask = filtered['risk_group'] == group
        if mask.sum() < 5: continue
        kmf = KaplanMeierFitter()
        kmf.fit(filtered.loc[mask, 'time'], event_observed=filtered.loc[mask, 'cid'],
                label=f'{group} (n={mask.sum()})')
        kmf.plot_survival_function(ax=axes[2], color=color, ci_show=False)
    axes[2].set_title('Survival by Transmission Risk Group\n(1996 CDC Classification)')
    axes[2].set_xlabel('Days')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Tab 4: Equity Analysis ────────────────────────────────────────────────────
with tab4:
    st.subheader("Model Equity Analysis")
    st.markdown("Does the Random Forest model perform equally well across demographic groups?")

    eq_col1, eq_col2 = st.columns(2)

    with eq_col1:
        st.markdown("**AUC by Race**")
        st.caption("⚠️ Race recorded as binary in the 1996 trial protocol — a dataset limitation, not our classification choice.")
        for race_val, label in [(0, 'White'), (1, 'Non-white')]:
            mask = df['race'] == race_val
            X_sub = df[feature_cols][mask]
            y_sub = df['cd4_improved'][mask]
            auc = roc_auc_score(y_sub, rf.predict_proba(X_sub)[:, 1])
            st.metric(f"{label} (n={mask.sum()})", f"AUC = {auc:.4f}")

    with eq_col2:
        st.markdown("**AUC by Gender**")
        for gender_val, label in [(0, 'Female'), (1, 'Male')]:
            mask = df['gender'] == gender_val
            X_sub = df[feature_cols][mask]
            y_sub = df['cd4_improved'][mask]
            auc = roc_auc_score(y_sub, rf.predict_proba(X_sub)[:, 1])
            st.metric(f"{label} (n={mask.sum()})", f"AUC = {auc:.4f}")

    st.markdown("---")
    st.info("**Finding:** No algorithmic bias detected by race (AUC within 0.002). Gender gap (Female 0.920 vs Male 0.868) suggests differential treatment response patterns worth further clinical investigation.")

st.markdown("---")
st.caption("ImmuneEQ · Rice Datathon 2026 · Emmanuel Uzoma · Data: AIDS Clinical Trials Group Study 175, UCI ML Repository")
