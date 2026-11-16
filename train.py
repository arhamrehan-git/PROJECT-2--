
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

HERE = Path(__file__).parent
ART = HERE / "artifacts"
TARGET = "is_bad"

# The 5 signals the model uses to decide approve / reject.
# Age, employment length etc. are deliberately left out (age is a protected attribute in credit decisions).
NUMERIC = ["person_income", "loan_amnt", "loan_int_rate"]
CATEGORICAL = ["person_home_ownership", "cb_person_default_on_file"]
FEATURES = NUMERIC + CATEGORICAL
HOME_TYPES = ["RENT", "MORTGAGE", "OWN", "OTHER"]
LABELS = {"person_income": "Annual income", "loan_amnt": "Loan amount", "loan_int_rate": "Interest rate",
          "person_home_ownership": "Home ownership", "cb_person_default_on_file": "Prior default on file"}


def load_data(path=HERE / "credit_risk_dataset.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    out = df[FEATURES].copy()
    out[TARGET] = df["loan_status"].astype(int)
    return out


def build_model() -> Pipeline:
    """One-hot the 2 categorical fields, pass the numbers through (gradient boosting handles
    missing values and skewed money fields natively, so no scaling or imputing is needed)."""
    prep = ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    clf = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                                         early_stopping=True, random_state=42)
    return Pipeline([("prep", prep), ("clf", clf)])


def best_threshold(y, p) -> float:
    """Youden's J: cut-off maximising (true-positive rate - false-positive rate)."""
    ts = np.linspace(0.02, 0.9, 89)
    j = [((p >= t) & (y == 1)).sum() / (y == 1).sum() - ((p >= t) & (y == 0)).sum() / (y == 0).sum() for t in ts]
    return float(ts[int(np.argmax(j))])


# ---------------- explainability (only needs model.predict_proba)
def permutation_importance(model, X, y, n_repeats=3, seed=0) -> pd.Series:
    """GLOBAL: shuffle one column at a time; the bigger the AUC drop, the more the model relies on it."""
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, model.predict_proba(X)[:, 1])
    out = {}
    for col in FEATURES:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy(); Xp[col] = rng.permutation(Xp[col].values)
            drops.append(base - roc_auc_score(y, model.predict_proba(Xp)[:, 1]))
        out[col] = float(np.mean(drops))
    return pd.Series(out).sort_values(ascending=False)


def reference_profile(train_df: pd.DataFrame) -> dict:
    """A 'typical applicant' (median / most common values) used as the comparison baseline."""
    return {c: (train_df[c].median() if c in NUMERIC else train_df[c].mode()[0]) for c in FEATURES}


def local_explanation(model, applicant: pd.DataFrame, reference: dict) -> pd.Series:
    """LOCAL: swap each field for the typical value and see how this applicant's risk changes.
    > 0 pushes risk up vs. a typical applicant, < 0 pulls it down."""
    p = model.predict_proba(applicant)[0, 1]
    deltas = {}
    for col in FEATURES:
        alt = applicant.copy(); alt[col] = reference[col]
        deltas[col] = p - model.predict_proba(alt)[0, 1]
    return pd.Series(deltas).sort_values(key=abs, ascending=False)


def main():
    ART.mkdir(exist_ok=True)
    df = load_data()
    # 60/20/20: train / validation (pick decision threshold) / test (held out). Stratified.
    train_df, tmp = train_test_split(df, test_size=.4, stratify=df[TARGET], random_state=42)
    val_df, test_df = train_test_split(tmp, test_size=.5, stratify=tmp[TARGET], random_state=42)
    X_tr, y_tr = train_df[FEATURES], train_df[TARGET]
    X_val, y_val = val_df[FEATURES], val_df[TARGET]
    X_te, y_te = test_df[FEATURES], test_df[TARGET]

    model = build_model().fit(X_tr, y_tr)
    p_te = model.predict_proba(X_te)[:, 1]
    thr = best_threshold(y_val.values, model.predict_proba(X_val)[:, 1])

    joblib.dump(model, ART / "model.joblib")
    joblib.dump({"y_test": y_te.values, "proba": p_te, "threshold": thr,
                 "loan_test": X_te["loan_amnt"].values,  # used by the app's profit simulation
                 "bad_rate": float(df[TARGET].mean()),
                 "importance": permutation_importance(model, X_te, y_te),
                 "reference": reference_profile(train_df)}, ART / "report.joblib")
    print(f"Decision threshold {thr:.2f} | test AUC {roc_auc_score(y_te, p_te):.3f} | saved to {ART}")


if __name__ == "__main__":
    main()
