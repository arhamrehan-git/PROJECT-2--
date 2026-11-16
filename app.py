"""app.py - Streamlit UI.  Run:  streamlit run app.py   (after  python train.py)"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from train import FEATURES, HOME_TYPES, LABELS, local_explanation

ART = Path(__file__).parent / "artifacts"
TRAIN_MAX_LOAN = 35_000  # largest loan in the training data
GOOD, BAD, ACCENT, GRAY = "#2e9e5b", "#d9534f", "#3b82f6", "#9aa0a6"

st.set_page_config(page_title="Loan Risk AI", page_icon="🏦", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero h1 {margin: 0; font-size: 2.1rem;}
.hero p {margin: .2rem 0 1rem 0; opacity: .7;}
.card {border: 1px solid rgba(128,128,128,.28); background: rgba(128,128,128,.07);
       border-radius: 14px; padding: 16px 18px; height: 100%;}
.card .lbl {font-size: .78rem; opacity: .7; text-transform: uppercase; letter-spacing: .04em;}
.card .val {font-size: 1.9rem; font-weight: 700; line-height: 1.25;}
.card .sub {font-size: .8rem; opacity: .65;}
.verdict {border-radius: 14px; padding: 18px 20px; font-weight: 700; font-size: 1.5rem; color: #fff;}
.verdict small {display: block; font-weight: 400; font-size: .85rem; opacity: .9; margin-top: 2px;}
.offer {border-radius: 14px; padding: 14px 18px; margin-bottom: 10px; border: 1px solid;}
.offer b {font-size: 1.05rem;}
.meter {position: relative; height: 16px; border-radius: 8px; margin: 26px 0 6px 0;
        background: linear-gradient(90deg, #2e9e5b 0%, #e8b923 40%, #d9534f 100%);}
.meter .mark {position: absolute; top: -6px; width: 4px; height: 28px; background: currentColor; border-radius: 2px;}
.meter .tag {position: absolute; top: -24px; font-size: .72rem; transform: translateX(-50%); white-space: nowrap;}
.scale {display: flex; justify-content: space-between; font-size: .72rem; opacity: .6;}
</style>
""", unsafe_allow_html=True)

# charts: transparent + mid-grey text so they look right in both light and dark themes
plt.rcParams.update({
    "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none", "savefig.transparent": True,
    "text.color": GRAY, "axes.labelcolor": GRAY, "xtick.color": GRAY, "ytick.color": GRAY, "axes.edgecolor": GRAY,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.grid": True, "grid.color": "#80808030", "grid.linewidth": .6, "axes.axisbelow": True,
    "legend.frameon": False})


@st.cache_resource
def load():
    return joblib.load(ART / "model.joblib"), joblib.load(ART / "report.joblib")


try:
    model, report = load()
except FileNotFoundError:
    st.error("No trained model found. Run `python train.py` first."); st.stop()


def card(label, value, sub="", color=None):
    style = f' style="color:{color}"' if color else ""
    return f'<div class="card"><div class="lbl">{label}</div><div class="val"{style}>{value}</div><div class="sub">{sub}</div></div>'


def sweep_loan(row, amounts):
    """Risk of the same applicant at many loan amounts (one batch prediction)."""
    grid = pd.concat([row] * len(amounts), ignore_index=True)
    grid["loan_amnt"] = amounts
    return model.predict_proba(grid)[:, 1]


# ---------------- sidebar: bank policy, loan economics, demo profiles
PRESETS = {
    "Custom": None,
    "Safe profile": dict(income=90_000, loan=6_000, rate=7.5, home="MORTGAGE", dflt="N"),
    "Loan too large": dict(income=40_000, loan=15_000, rate=13.0, home="RENT", dflt="N"),
    "Risky profile": dict(income=22_000, loan=15_000, rate=18.5, home="RENT", dflt="Y"),
}
DEFAULTS = dict(income=60_000, loan=10_000, rate=11.0, home="RENT", dflt="N", gain=10.0, loss=60.0)
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def apply_preset():
    p = PRESETS[st.session_state["preset"]]
    if p:
        st.session_state.update(p)


with st.sidebar:
    st.header("💰 Loan economics")
    st.number_input("Profit if repaid (% of loan)", 0.0, 100.0, step=1.0, key="gain",
                    help="Interest and fees the bank earns when the loan is repaid.")
    st.number_input("Loss if defaulted (% of loan)", 0.0, 100.0, step=5.0, key="loss",
                    help="Share of the loan the bank loses when the borrower defaults.")
    g, l = st.session_state["gain"] / 100, st.session_state["loss"] / 100
    breakeven = min(max(g / (g + l), 0.01), 0.95) if (g + l) > 0 else 0.5

    st.divider()
    st.header("🏦 Bank policy")
    rule = st.radio("Decision rule", ["Risk cut-off (slider)", "Maximise expected profit"], key="rule",
                    help="The profit rule approves a loan only when its expected profit is positive.")
    if rule.startswith("Risk"):
        thr = st.slider("Reject if risk is above (%)", 2, 90, int(report["threshold"] * 100)) / 100
        st.caption("Default is tuned on validation data. **Lower = stricter bank.** "
                   "Changing it does not retrain the model.")
    else:
        thr = breakeven
        st.info(f"Break-even risk: **{thr:.1%}**. Loans riskier than this lose money on average.")

    st.divider()
    st.header("🧪 Demo profiles")
    st.selectbox("Load an example applicant", list(PRESETS), key="preset", on_change=apply_preset)

st.markdown('<div class="hero"><h1>🏦 Loan Risk AI</h1>'
            '<p>Predicts the chance a borrower defaults, decides against your bank policy, '
            'and shows what it would take to change the outcome.</p></div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["💳 Loan Calculator", "💰 Business Impact", "📊 Model & Pipeline"])

# ======================= TAB 1: calculator
with tab1:
    left, right = st.columns([5, 6], gap="large")
    with left:
        st.subheader("Applicant details")
        with st.container(border=True):
            st.number_input("Annual income", 1_000, 10_000_000, step=1_000, key="income")
            st.number_input("Loan amount", 500, 100_000, step=500, key="loan")
            st.number_input("Interest rate (%)", 0.0, 40.0, step=0.25, key="rate")
            c1, c2 = st.columns(2)
            c1.selectbox("Home ownership", HOME_TYPES, key="home", format_func=str.title)
            c2.radio("Prior default on file?", ["N", "Y"], key="dflt", horizontal=True,
                     format_func=lambda v: {"N": "No", "Y": "Yes"}[v],
                     help="Credit-bureau flag: has this person ever defaulted on a loan before?")
        if st.session_state["loan"] > TRAIN_MAX_LOAN:
            st.warning(f"Loan is above {TRAIN_MAX_LOAN:,}, the largest in the training data. "
                       "The score is less reliable outside the range the model has seen.")

    loan = st.session_state["loan"]
    row = pd.DataFrame([{"person_income": st.session_state["income"], "loan_amnt": loan,
                         "loan_int_rate": st.session_state["rate"],
                         "person_home_ownership": st.session_state["home"],
                         "cb_person_default_on_file": st.session_state["dflt"]}])[FEATURES]
    risk = float(model.predict_proba(row)[0, 1])
    approved = risk <= thr
    ep = loan * ((1 - risk) * g - risk * l)  # expected profit if this loan were approved

    with right:
        st.subheader("Decision")
        bg = GOOD if approved else BAD
        verdict = "APPROVED" if approved else "REJECTED: HIGH RISK"
        note = f"{abs(risk - thr) * 100:.1f} points {'below' if approved else 'above'} the {thr:.0%} cut-off"
        st.markdown(f'<div class="verdict" style="background:{bg}">{verdict}<small>{note}</small></div>',
                    unsafe_allow_html=True)
        st.write("")
        m1, m2, m3 = st.columns(3)
        m1.markdown(card("Risk score", f"{risk:.1%}", "probability of default"), unsafe_allow_html=True)
        m2.markdown(card("Cut-off", f"{thr:.0%}", "bank policy"), unsafe_allow_html=True)
        m3.markdown(card("Expected profit", f"{ep:+,.0f}", "if approved", GOOD if ep >= 0 else BAD),
                    unsafe_allow_html=True)

        rp, tp_ = min(risk, 1.0) * 100, thr * 100
        st.markdown(
            f'<div class="meter"><div class="tag" style="left:{rp}%"><b>Risk {risk:.0%}</b></div>'
            f'<div class="mark" style="left:calc({rp}% - 2px)"></div>'
            f'<div class="tag" style="left:{tp_}%;top:22px;opacity:.7">cut-off</div>'
            f'<div class="mark" style="left:calc({tp_}% - 2px);opacity:.35;height:20px;top:-2px"></div></div>'
            '<div class="scale"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>',
            unsafe_allow_html=True)
        if approved and ep < 0:
            st.warning("Approved by the cut-off, but expected profit is negative with your loan economics.")
        elif not approved and ep > 0:
            st.info("Rejected by the cut-off, although expected profit is positive. "
                    "The profit rule in the sidebar would approve this loan.")

    st.divider()
    why, offer = st.columns(2, gap="large")

    # ---- why this decision
    with why:
        st.subheader("Why this decision?")
        expl_all = local_explanation(model, row, report["reference"])
        up, down = expl_all[expl_all > 0], expl_all[expl_all < 0]
        parts = []
        if len(up):
            parts.append(f"**{LABELS[up.idxmax()]}** pushes risk up the most (+{up.max() * 100:.1f} points)")
        if len(down):
            parts.append(f"**{LABELS[down.idxmin()]}** pulls it down the most ({down.min() * 100:.1f} points)")
        st.markdown("Compared with a typical applicant: " + ", while ".join(parts) + "." if parts
                    else "This applicant looks like a typical borrower.")
        expl = expl_all[::-1]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh([LABELS[i] for i in expl.index], expl.values * 100,
                color=[BAD if v > 0 else GOOD for v in expl.values])
        ax.axvline(0, color=GRAY, lw=.8); ax.grid(axis="y", visible=False)
        ax.set_xlabel("Change in risk vs. a typical applicant (points)")
        fig.tight_layout(); st.pyplot(fig)
        st.caption("🟥 pushes risk up   🟩 pulls risk down")

    # ---- what would change this decision
    with offer:
        st.subheader("What would change this decision?")
        top = max(TRAIN_MAX_LOAN, loan)
        amounts = np.arange(500, top + 1, 250)
        risks = sweep_loan(row, amounts)
        ok = risks <= thr
        best = None
        if not approved:
            lower = amounts[(amounts < loan) & ok]
            if len(lower):
                best = int(lower.max())
                new_risk = float(risks[amounts == best][0])
                st.markdown(
                    f'<div class="offer" style="border-color:{GOOD};background:{GOOD}22">'
                    f'<b>Counter-offer: approve up to {best:,}</b><br>'
                    f'Cutting the loan from {loan:,} to {best:,} lowers the risk from {risk:.1%} to {new_risk:.1%}, '
                    f'under the {thr:.0%} cut-off.</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="offer" style="border-color:{BAD};background:{BAD}22">'
                    f'<b>No loan amount would be approved</b><br>'
                    f'Even a small loan stays above the cut-off, so the blockers are the applicant\'s income, '
                    f'home status, prior default or interest rate, not the loan size.</div>', unsafe_allow_html=True)
        else:
            higher = amounts > loan
            blocked = amounts[higher & ~ok]
            if len(blocked):
                limit = int(blocked.min() - 250)
                msg = f"Could borrow up to <b>{limit:,}</b> before the risk crosses the {thr:.0%} cut-off."
            else:
                msg = f"Stays approved at every loan size up to {top:,}."
            st.markdown(f'<div class="offer" style="border-color:{GOOD};background:{GOOD}22">'
                        f'<b>Headroom</b><br>{msg}</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(amounts, risks * 100, color=ACCENT, lw=2)
        ax.axhline(thr * 100, color=BAD, ls="--", lw=1.2, label=f"cut-off {thr:.0%}")
        ax.fill_between(amounts, 0, thr * 100, color=GOOD, alpha=.12)
        ax.scatter([loan], [risk * 100], color=GRAY, s=60, zorder=5, label="current loan")
        if best:
            ax.scatter([best], [float(risks[amounts == best][0]) * 100], color=GOOD, s=70, zorder=6,
                       label="counter-offer")
        ax.set_ylim(0, min(100, max(risks.max() * 100 + 5, thr * 100 + 10)))
        ax.set(xlabel="Loan amount", ylabel="Risk of default (%)"); ax.legend(loc="upper left")
        fig.tight_layout(); st.pyplot(fig)
        st.caption("Only the loan amount is treated as adjustable. Income, home status and history cannot be "
                   "changed by a counter-offer, and the interest rate reflects the risk rather than causing it. "
                   "Green zone = approved.")

# ======================= TAB 2: business impact
with tab2:
    y, p = report["y_test"], report["proba"]
    if "loan_test" not in report:
        st.info("Re-run `python train.py` to enable the profit simulation (the report needs loan amounts).")
    else:
        L = report["loan_test"].astype(float)
        unit = np.where(y == 0, g, -l) * L  # profit of each test loan if it were approved

        def outcome(t):
            m = p <= t
            return float(unit[m].sum()), float(m.mean()), int((m & (y == 1)).sum())

        ts = np.linspace(0, 0.9, 91)
        profits = np.array([outcome(t)[0] for t in ts])
        t_best = float(ts[int(np.argmax(profits))])
        cur_profit, cur_appr, cur_def = outcome(thr)
        all_profit = outcome(1.0)[0]
        best_profit = float(profits.max())

        st.subheader("What each policy is worth in money")
        st.caption(f"Simulated on {len(y):,} held-out loans the model never saw, with your loan economics "
                   f"({g:.0%} profit if repaid, {l:.0%} loss if defaulted). Change them in the sidebar.")
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(card("Profit at current cut-off", f"{cur_profit:,.0f}", f"cut-off {thr:.0%}",
                         GOOD if cur_profit >= 0 else BAD), unsafe_allow_html=True)
        k2.markdown(card("Best achievable profit", f"{best_profit:,.0f}", f"at a {t_best:.0%} cut-off", GOOD),
                    unsafe_allow_html=True)
        k3.markdown(card("Approve everyone", f"{all_profit:,.0f}", "no model at all",
                         GOOD if all_profit >= 0 else BAD), unsafe_allow_html=True)
        k4.markdown(card("Gain from the model", f"{cur_profit - all_profit:+,.0f}", "vs. approving everyone"),
                    unsafe_allow_html=True)
        st.write("")

        c1, c2 = st.columns([3, 2], gap="large")
        with c1:
            st.markdown("**Profit vs. cut-off**")
            fig, ax = plt.subplots(figsize=(7, 3.6))
            ax.plot(ts * 100, profits, color=ACCENT, lw=2)
            ax.axhline(0, color=GRAY, lw=.8)
            ax.axvline(thr * 100, color=BAD, ls="--", lw=1.2, label=f"current ({thr:.0%})")
            ax.scatter([t_best * 100], [best_profit], color=GOOD, s=80, zorder=5, label=f"best ({t_best:.0%})")
            ax.set(xlabel="Cut-off: approve loans with risk up to (%)", ylabel="Total profit"); ax.legend()
            fig.tight_layout(); st.pyplot(fig)
        with c2:
            st.markdown("**Policy comparison**")
            pol = [("Approve everyone", 1.0), ("Statistical cut-off", float(report["threshold"])),
                   ("Profit-optimal cut-off", t_best), ("Your current setting", thr)]
            rows = []
            for name, t in pol:
                pr, ap, df_ = outcome(t)
                rows.append({"Policy": name, "Cut-off": f"{min(t, 1):.0%}", "Approved": f"{ap:.0%}",
                             "Defaults let in": df_, "Profit": f"{pr:,.0f}"})
            st.dataframe(pd.DataFrame(rows).set_index("Policy"))
            st.caption("The best statistical balance (Youden's J) is not the same as the most profitable "
                       "cut-off, because a missed default usually costs far more than a lost good customer.")

# ======================= TAB 3: model performance + pipeline
with tab3:
    y, p = report["y_test"], report["proba"]
    pred = p >= thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    recall, fpr_ = tp / (tp + fn), fp / (fp + tn)
    precision = tp / (tp + fp) if (tp + fp) else 0.0

    st.caption(f"Scores come from a held-out test set of {len(y):,} loans the model never saw, "
               f"at the current cut-off of {thr:.0%}.")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(card("ROC-AUC", f"{roc_auc_score(y, p):.3f}", "ranking quality (1.0 = perfect)"), unsafe_allow_html=True)
    k2.markdown(card("Bad loans caught", f"{recall:.1%}", "recall"), unsafe_allow_html=True)
    k3.markdown(card("Good loans wrongly flagged", f"{fpr_:.1%}", "false positive rate"), unsafe_allow_html=True)
    k4.markdown(card("Flags that were right", f"{precision:.1%}", "precision"), unsafe_allow_html=True)
    st.write("")

    l_, r_ = st.columns(2, gap="large")
    with l_:
        st.markdown("**ROC curve**")
        fpr, tpr, _ = roc_curve(y, p)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color=ACCENT, lw=2); ax.fill_between(fpr, tpr, alpha=.12, color=ACCENT)
        ax.plot([0, 1], [0, 1], "--", c=GRAY)
        ax.scatter([fpr_], [recall], color=BAD, zorder=5, label=f"current cut-off ({thr:.0%})")
        ax.set(xlabel="False positive rate", ylabel="True positive rate"); ax.legend(loc="lower right")
        fig.tight_layout(); st.pyplot(fig)
    with r_:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y, pred)
        fig, ax = plt.subplots(figsize=(5, 4)); ax.imshow(cm, cmap="Blues"); ax.grid(False)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=14,
                    color="white" if v > cm.max() / 2 else "#111")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Good", "Bad"], yticklabels=["Good", "Bad"],
               xlabel="Predicted", ylabel="Actual")
        fig.tight_layout(); st.pyplot(fig)

    st.markdown("**What drives the model overall?**")
    imp = report["importance"][::-1]
    fig, ax = plt.subplots(figsize=(8, 2.8)); ax.barh([LABELS[i] for i in imp.index], imp.values, color=ACCENT)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Drop in AUC when this feature is shuffled (bigger = more important)")
    fig.tight_layout(); st.pyplot(fig)

    with st.expander("📘 How the pipeline works"):
        a, b = st.columns(2, gap="large")
        with a:
            st.markdown("""
1. **Data:** ~32,000 past loans, each labelled defaulted or repaid.
2. **Split 60/20/20:** train (learn), validation (pick cut-off), test (honest score).
3. **Model:** gradient boosting, many small decision trees where each corrects the previous one.
4. **Cut-off:** chosen on validation data, or by expected profit with your loan economics.
5. **Decision:** risk above the cut-off means reject, otherwise approve.
6. **Explanation:** every decision is compared with a typical applicant, one input at a time.
""")
        with b:
            st.table(pd.DataFrame({"Input": list(LABELS.values()),
                                   "Meaning": ["Yearly income of the applicant", "Amount requested",
                                               "Rate charged on the loan", "Rent, mortgage, own or other",
                                               "Has the person ever defaulted before (credit-bureau flag)"]}
                                  ).set_index("Input"))